from dotenv import load_dotenv
import os
import json
from datetime import datetime
 
load_dotenv()
 
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
 
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
 
try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
 
 
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
        faiss_results  = db.similarity_search_with_score(query, k=k)
        faiss_docs     = [r[0] for r in faiss_results]
        faiss_scores   = [r[1] for r in faiss_results]
        similarities   = [round((1 / (1 + s)) * 100, 1) for s in faiss_scores]
        avg_confidence = round(sum(similarities) / len(similarities), 1)
    except:
        faiss_docs     = db.similarity_search(query, k=k)
        avg_confidence = 65.0
 
    if BM25_AVAILABLE and bm25_index is not None and all_docs:
        tokenized_query = query.lower().split()
        bm25_scores     = bm25_index.get_scores(tokenized_query)
        top_indices     = sorted(range(len(bm25_scores)),
                                 key=lambda i: bm25_scores[i], reverse=True)[:k]
        bm25_docs  = [all_docs[i] for i in top_indices]
        combined   = reciprocal_rank_fusion(faiss_docs, bm25_docs)[:k]
        search_type = "hybrid"
    else:
        combined    = faiss_docs
        search_type = "semantic"
 
    return combined, avg_confidence, search_type
 
 
def retrieve_with_scores(db, query, k=3):
    try:
        results       = db.similarity_search_with_score(query, k=k)
        docs          = [r[0] for r in results]
        scores        = [r[1] for r in results]
        similarities  = [round((1 / (1 + s)) * 100, 1) for s in scores]
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
 
    prompt = f"""You are a secure AI Study Assistant.
Rules:
- Answer ONLY from the provided context
- If not found say: "This topic is not covered in the uploaded material."
- Be educational and clear
 
{"Previous conversation:" + chr(10) + history_str + chr(10) if history_str else ""}
 
Context:
{context}
 
Question: {query}
 
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
    context      = " ".join([d.page_content.lower() for d in docs])
    answer_lower = answer.lower()
    query_lower  = query.lower()
 
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
    query_words   = set(query_lower.split()) - stop_words
    if query_words:
        overlap = len(query_words & context_words)
        context_score = round(min((overlap / len(query_words)) * 100, 100), 1)
    else:
        context_score = 70.0
 
    return {
        "faithfulness":      min(faithfulness_score, 98.0),
        "answer_relevancy":  min(relevancy_score, 98.0),
        "context_precision": min(context_score, 98.0),
        "method": "local"
    }
 
 
def full_ragas_evaluate(query, answer, docs):
    if not RAGAS_AVAILABLE:
        return local_evaluate(query, answer, docs)
    try:
        contexts = [doc.page_content for doc in docs]
        data = {
            "question": [query], "answer": [answer],
            "contexts": [contexts], "ground_truth": [answer]
        }
        dataset = Dataset.from_dict(data)
        result  = ragas_eval(dataset,
                             metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness":      round(result["faithfulness"] * 100, 1),
            "answer_relevancy":  round(result["answer_relevancy"] * 100, 1),
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
# TIER 1 — FLASHCARD GENERATOR
# ───────────────────────────────────────────────
def generate_flashcards(query, answer, docs, num_cards=5):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are an expert study coach creating flashcards.
Create {num_cards} flashcards based on the question and answer.
 
Return ONLY a valid JSON array:
[
  {{
    "front": "Short concept or question (max 15 words)",
    "back": "Clear concise explanation (max 50 words)",
    "topic": "Topic name"
  }}
]
 
Question: {query}
Answer: {answer[:1000]}
Context: {context[:500]}"""
 
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
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return []
 
 
# ───────────────────────────────────────────────
# TIER 1 — DOCUMENT INTELLIGENCE
# ───────────────────────────────────────────────
def analyze_document(docs):
    sample  = docs[:20] if len(docs) > 20 else docs
    context = "\n\n".join([d.page_content for d in sample])
    total_pages = max([d.metadata.get("page", 0) for d in docs]) + 1 if docs else 1
 
    prompt = f"""Analyze this study material and return a structured report.
Return ONLY a valid JSON object:
{{
  "summary": "2-3 sentence overview",
  "key_topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"],
  "difficulty": "Beginner",
  "estimated_hours": 3,
  "total_concepts": 12,
  "suggested_question": "Specific question about this material"
}}
 
difficulty must be: "Beginner", "Intermediate", or "Advanced"
 
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
        result = json.loads(raw)
        result["total_pages"]  = total_pages
        result["total_chunks"] = len(docs)
        return result
    except:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                result = json.loads(raw[start:end])
                result["total_pages"]  = total_pages
                result["total_chunks"] = len(docs)
                return result
            except:
                pass
        return {
            "summary": "Document processed successfully.",
            "key_topics": ["Core Concepts"],
            "difficulty": "Intermediate",
            "estimated_hours": 3,
            "total_concepts": len(docs),
            "suggested_question": "What are the main topics covered?",
            "total_pages": total_pages,
            "total_chunks": len(docs)
        }
 
 
# ───────────────────────────────────────────────
# TIER 1 — VOICE TRANSCRIPTION
# ───────────────────────────────────────────────
def transcribe_audio(audio_bytes):
    import openai
    import tempfile
 
    client = openai.OpenAI(api_key=api_key)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
 
    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        return f"Transcription failed: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
 
 
# ───────────────────────────────────────────────
# TIER 2 — ADAPTIVE MCQ GENERATION ← NEW
# Generates questions based on difficulty level
# Easy: simple recall, hints in options
# Medium: application-level questions
# Hard: analysis, tricky distractors
# ───────────────────────────────────────────────
def generate_adaptive_mcqs(docs, num_questions=5, difficulty="Medium"):
    context = "\n\n".join([d.page_content for d in docs])
 
    difficulty_instructions = {
        "Easy": """
- Ask simple recall and definition questions
- Use clear, straightforward language
- Make wrong options obviously different from correct answer
- Focus on basic concepts and definitions
- Add a small hint in the question itself""",
 
        "Medium": """
- Ask application and understanding questions
- Require students to apply concepts, not just recall
- Make distractors plausible but clearly wrong on closer inspection
- Mix conceptual and applied questions
- Test relationships between concepts""",
 
        "Hard": """
- Ask analysis, evaluation and synthesis questions
- Create tricky distractors that are partially correct
- Include edge cases and exceptions
- Require deep understanding and critical thinking
- Test ability to distinguish between similar concepts
- Use scenario-based questions"""
    }
 
    instruction = difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])
 
    prompt = f"""You are an expert exam creator making a {difficulty.upper()} difficulty quiz.
 
Difficulty guidelines for {difficulty}:
{instruction}
 
Generate {num_questions} MCQs from the context below.
Return ONLY a valid JSON array, no extra text:
 
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Why this answer is correct",
    "difficulty": "{difficulty}",
    "hint": "Small hint for the student (for Easy only, else empty string)"
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
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return []
 
 
# Keep old function for backward compat
def generate_mcqs(docs, num_questions=5):
    return generate_adaptive_mcqs(docs, num_questions, "Medium")
 
 
# ───────────────────────────────────────────────
# TIER 2 — CONCEPT MAP GENERATOR ← NEW
# Generates topic relationships for network graph
# ───────────────────────────────────────────────
def generate_concept_map(docs):
    """
    Analyzes PDF and generates concept relationships.
    Returns nodes and edges for network visualization.
    """
    sample  = docs[:15] if len(docs) > 15 else docs
    context = "\n\n".join([d.page_content for d in sample])
 
    prompt = f"""You are an academic knowledge graph expert.
 
Analyze this study material and identify the key concepts and how they relate to each other.
 
Return ONLY a valid JSON object:
{{
  "nodes": [
    {{"id": "node1", "label": "Machine Learning", "category": "main"}},
    {{"id": "node2", "label": "Supervised Learning", "category": "sub"}},
    {{"id": "node3", "label": "Decision Trees", "category": "detail"}}
  ],
  "edges": [
    {{"source": "node1", "target": "node2", "relationship": "includes"}},
    {{"source": "node2", "target": "node3", "relationship": "uses"}}
  ]
}}
 
Rules:
- 8-15 nodes total
- category must be: "main" (core topic), "sub" (subtopic), "detail" (specific concept)
- relationship should be a short verb: "includes", "uses", "requires", "leads to", "part of", "related to"
- Make sure every node has at least one edge
 
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
        return json.loads(raw)
    except:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return None
 
 
# ───────────────────────────────────────────────
# TIER 2 — EXAM MODE ← NEW
# Generates full timed exam with MCQ + short answer
# ───────────────────────────────────────────────
def generate_exam(docs, num_mcq=10, num_short=3, difficulty="Medium"):
    """
    Generates a complete exam paper with:
    - MCQ questions (adaptive difficulty)
    - Short answer questions
    Returns structured JSON exam paper
    """
    context = "\n\n".join([d.page_content for d in docs[:15]])
 
    prompt = f"""You are an expert exam paper setter creating a {difficulty} difficulty exam.
 
Generate a complete exam with:
- {num_mcq} MCQ questions
- {num_short} short answer questions (2-5 marks each)
 
Return ONLY a valid JSON object:
{{
  "exam_title": "Mock Exam: [Subject]",
  "difficulty": "{difficulty}",
  "total_marks": {num_mcq + (num_short * 5)},
  "mcqs": [
    {{
      "id": 1,
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "marks": 1,
      "explanation": "..."
    }}
  ],
  "short_answers": [
    {{
      "id": 1,
      "question": "Explain ... in detail",
      "marks": 5,
      "key_points": ["Point 1", "Point 2", "Point 3"],
      "model_answer": "Brief model answer here"
    }}
  ]
}}
 
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
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return None
 
 
# ───────────────────────────────────────────────
# STUDY PLAN
# ───────────────────────────────────────────────
def generate_study_plan(docs, num_days=7, weak_topics=None):
    sample  = docs[:15] if len(docs) > 15 else docs
    context = "\n\n".join([d.page_content for d in sample])
 
    weak_topics_str = ""
    if weak_topics:
        top_weak = sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_list = [t[0] for t in top_weak]
        weak_topics_str = f"\nIMPORTANT: Prioritize these weak topics: {weak_list}\n"
 
    prompt = f"""You are an expert academic planner creating a personalized {num_days}-day study plan.
{weak_topics_str}
 
Return ONLY a valid JSON object:
{{
  "plan_title": "{num_days}-Day Study Plan: [Subject]",
  "total_days": {num_days},
  "days": [
    {{
      "day": 1,
      "title": "Day title",
      "topics": ["Topic 1", "Topic 2"],
      "study_focus": "One sentence focus",
      "suggested_questions": ["Q1?", "Q2?", "Q3?"],
      "difficulty": "Easy",
      "estimated_hours": 2
    }}
  ]
}}
 
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
        return json.loads(raw)
    except:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
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