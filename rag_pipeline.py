from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from ragas import evaluate as ragas_eval
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


# ───────────────────────────────────────────────
# PDF + CHUNKING
# ───────────────────────────────────────────────
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


# ───────────────────────────────────────────────
# VECTOR DB
# ───────────────────────────────────────────────
def create_db(docs):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.from_documents(docs, embeddings)


# ───────────────────────────────────────────────
# BM25 INDEX
# ───────────────────────────────────────────────
def build_bm25_index(docs):
    if not BM25_AVAILABLE:
        return None, docs
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


# ───────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ───────────────────────────────────────────────
def reciprocal_rank_fusion(faiss_docs, bm25_docs, k=60):
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + (1 / (k + rank + 1))
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:120]
        scores[key] = scores.get(key, 0) + (1 / (k + rank + 1))
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ───────────────────────────────────────────────
# HYBRID RETRIEVAL
# ───────────────────────────────────────────────
def hybrid_retrieve(db, bm25_index, all_docs, query, k=4):
    try:
        faiss_results = db.similarity_search_with_score(query, k=k)
        faiss_docs = [r[0] for r in faiss_results]
        faiss_scores = [r[1] for r in faiss_results]
        similarities = [round((1 / (1 + s)) * 100, 1) for s in faiss_scores]
        avg_confidence = round(sum(similarities) / len(similarities), 1)
    except:
        faiss_docs = db.similarity_search(query, k=k)
        avg_confidence = 65.0

    if BM25_AVAILABLE and bm25_index is not None and all_docs:
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:k]
        bm25_docs = [all_docs[i] for i in top_indices]
        combined = reciprocal_rank_fusion(faiss_docs, bm25_docs)[:k]
        search_type = "hybrid"
    else:
        combined = faiss_docs
        search_type = "semantic"

    return combined, avg_confidence, search_type


def retrieve_with_scores(db, query, k=3):
    try:
        results = db.similarity_search_with_score(query, k=k)
        docs = [r[0] for r in results]
        scores = [r[1] for r in results]
        similarities = [round((1 / (1 + s)) * 100, 1) for s in scores]
        avg_confidence = round(sum(similarities) / len(similarities), 1)
        return docs, avg_confidence, similarities
    except:
        docs = db.similarity_search(query, k=k)
        return docs, 65.0, [65.0] * k


# ───────────────────────────────────────────────
# SOURCE PAGES
# ───────────────────────────────────────────────
def get_sources(docs):
    pages = []
    for d in docs:
        if "page" in d.metadata:
            pages.append(d.metadata["page"] + 1)
    return sorted(list(set(pages)))


# ───────────────────────────────────────────────
# TOPIC DETECTION
# ───────────────────────────────────────────────
def detect_topic(query):
    prompt = f"""Classify this student question into ONE short academic topic (2-4 words max).
Examples: "Machine Learning", "Neural Networks", "Data Structures", "Operating Systems"

Question: {query}

Reply with ONLY the topic name, nothing else."""
    try:
        response = llm.invoke(prompt)
        topic = response.content.strip().strip('"').strip("'")
        return topic if topic else "General"
    except:
        return "General"


# ───────────────────────────────────────────────
# SECURITY
# ───────────────────────────────────────────────
BLOCKED_PHRASES = [
    "ignore previous", "ignore instructions", "system prompt",
    "api key", "password", "jailbreak", "forget rules",
    "act as", "you are now", "disregard", "override"
]

def is_safe_query(query):
    return not any(phrase in query.lower() for phrase in BLOCKED_PHRASES)


# ───────────────────────────────────────────────
# GENERATE ANSWER
# ───────────────────────────────────────────────
def generate_answer(query, docs, memory=None):
    context = "\n\n".join([d.page_content for d in docs])
    history_str = ""
    if memory:
        messages = memory.messages[-6:]
        history_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Assistant: {msg.content[:300]}...")
        history_str = "\n".join(history_lines)

    prompt = f"""You are a secure AI Study Assistant helping a student understand their study material.

Rules:
- Answer ONLY from the provided context
- If not found say: "This topic is not covered in the uploaded material."
- Be educational and clear

{"Previous conversation:" + chr(10) + history_str + chr(10) if history_str else ""}

Context from PDF:
{context}

Student Question: {query}

Provide a structured answer:
**2 Marks (brief):** ...
**5 Marks (detailed):** ...
**10 Marks (comprehensive):** ..."""

    response = llm.invoke(prompt)
    return response.content


# ───────────────────────────────────────────────
# LOCAL EVALUATION
# ───────────────────────────────────────────────
def local_evaluate(query, answer, docs):
    context = " ".join([d.page_content.lower() for d in docs])
    answer_lower = answer.lower()
    query_lower = query.lower()

    stop_words = {"what", "is", "the", "a", "an", "of", "in", "and",
                  "to", "how", "why", "when", "where", "which", "are",
                  "does", "do", "its", "it", "was", "be", "with", "for"}

    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    if sentences:
        grounded = sum(
            1 for s in sentences
            if any(word in context for word in s.lower().split() if len(word) > 4)
        )
        faithfulness_score = round((grounded / len(sentences)) * 100, 1)
    else:
        faithfulness_score = 70.0

    q_keywords = [w for w in query_lower.split()
                  if w not in stop_words and len(w) > 3]
    if q_keywords:
        matched = sum(1 for kw in q_keywords if kw in answer_lower)
        relevancy_score = round((matched / len(q_keywords)) * 100, 1)
    else:
        relevancy_score = 75.0

    context_words = set(context.split())
    query_words = set(query_lower.split()) - stop_words
    if query_words:
        overlap = len(query_words & context_words)
        context_score = round(min((overlap / len(query_words)) * 100, 100), 1)
    else:
        context_score = 70.0

    return {
        "faithfulness": min(faithfulness_score, 98.0),
        "answer_relevancy": min(relevancy_score, 98.0),
        "context_precision": min(context_score, 98.0),
        "method": "local"
    }


def full_ragas_evaluate(query, answer, docs):
    if not RAGAS_AVAILABLE:
        return local_evaluate(query, answer, docs)
    try:
        contexts = [doc.page_content for doc in docs]
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [answer]
        }
        dataset = Dataset.from_dict(data)
        result = ragas_eval(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision]
        )
        return {
            "faithfulness": round(result["faithfulness"] * 100, 1),
            "answer_relevancy": round(result["answer_relevancy"] * 100, 1),
            "context_precision": round(result["context_precision"] * 100, 1),
            "method": "ragas"
        }
    except:
        return local_evaluate(query, answer, docs)


def evaluate_answer(query, answer, docs):
    if RAGAS_AVAILABLE:
        return full_ragas_evaluate(query, answer, docs)
    return local_evaluate(query, answer, docs)


# ───────────────────────────────────────────────
# MCQ GENERATION
# ───────────────────────────────────────────────
def generate_mcqs(docs, num_questions=5):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are an AI exam creator. Generate {num_questions} MCQs from the context.
Return ONLY a valid JSON array, no extra text or markdown.

[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "..."
  }}
]

Context:
{context}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return []


# ───────────────────────────────────────────────
# EXTRACT KEY TOPICS FROM PDF  ← Day 2
# ───────────────────────────────────────────────
def extract_key_topics(docs):
    """
    Reads the full PDF content and extracts all key topics.
    Returns a list of topic strings.
    """
    # Use a sample of docs to avoid token limits
    sample_docs = docs[:15] if len(docs) > 15 else docs
    context = "\n\n".join([d.page_content for d in sample_docs])

    prompt = f"""You are an academic curriculum designer.

Read the study material below and extract ALL important topics covered in it.

Return ONLY a valid JSON array of topic strings. No extra text.
Each topic should be 2-5 words, specific and academic.

Example format:
["Supervised Learning", "K-Means Clustering", "Neural Networks", "Gradient Descent", "Overfitting"]

Study Material:
{context}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        topics = json.loads(raw)
        return [t for t in topics if isinstance(t, str)]
    except:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return ["Introduction", "Core Concepts", "Advanced Topics",
                "Applications", "Review", "Practice", "Revision"]


# ───────────────────────────────────────────────
# GENERATE 7-DAY STUDY PLAN  ← Day 2
# Personalizes based on weak topics detected
# ───────────────────────────────────────────────
def generate_study_plan(docs, num_days=7, weak_topics=None):
    """
    Generates a personalized study plan from PDF content.
    Prioritizes weak topics if detected from chat history.
    Returns a structured JSON plan.
    """
    sample_docs = docs[:15] if len(docs) > 15 else docs
    context = "\n\n".join([d.page_content for d in sample_docs])

    weak_topics_str = ""
    if weak_topics:
        top_weak = sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_list = [t[0] for t in top_weak]
        weak_topics_str = f"\nIMPORTANT: The student struggles with these topics, prioritize them: {weak_list}\n"

    prompt = f"""You are an expert academic planner creating a personalized {num_days}-day study plan.

{weak_topics_str}

Based on the study material below, create a {num_days}-day study plan.

Rules:
- Distribute topics logically (easy → hard progression)
- Each day should be manageable (2-3 hours max)
- Day 7 should always be revision/practice
- If weak topics exist, spread them across multiple days

Return ONLY a valid JSON object in exactly this format:
{{
  "plan_title": "7-Day Study Plan: [Subject Name]",
  "total_days": {num_days},
  "days": [
    {{
      "day": 1,
      "title": "Day title here",
      "topics": ["Topic 1", "Topic 2"],
      "study_focus": "One sentence describing what to focus on today",
      "suggested_questions": ["Question 1?", "Question 2?", "Question 3?"],
      "difficulty": "Easy",
      "estimated_hours": 2
    }}
  ]
}}

Difficulty must be one of: "Easy", "Medium", "Hard"
estimated_hours must be a number between 1 and 4

Study Material:
{context}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        plan = json.loads(raw)
        return plan
    except:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return None


# ───────────────────────────────────────────────
# MEMORY
# ───────────────────────────────────────────────
def create_memory():
    return ChatMessageHistory()
