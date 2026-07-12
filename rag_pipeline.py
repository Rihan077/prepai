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
 
 
# ── LIMITS ────────────────────────────────────────
MAX_PDFS        = 5
MAX_PDF_SIZE_MB = 20
MAX_PAGES       = 100
 
 
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
    scores  = {}
    doc_map = {}
    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0) + (1 / (k + rank + 1))
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:120]
        scores[key]  = scores.get(key, 0) + (1 / (k + rank + 1))
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]
 
 
# ───────────────────────────────────────────────
# SINGLE PDF HYBRID RETRIEVAL
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
        results        = db.similarity_search_with_score(query, k=k)
        docs           = [r[0] for r in results]
        scores         = [r[1] for r in results]
        similarities   = [round((1 / (1 + s)) * 100, 1) for s in scores]
        avg_confidence = round(sum(similarities) / len(similarities), 1)
        return docs, avg_confidence, similarities
    except:
        docs = db.similarity_search(query, k=k)
        return docs, 65.0, [65.0] * k
 
 
# ───────────────────────────────────────────────
# TIER 3 — MULTI-PDF INDEXING ← NEW
# ───────────────────────────────────────────────
def validate_pdf(file_bytes, filename):
    """
    Validates PDF before processing.
    Returns (is_valid, error_message)
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return False, f"{filename} is {size_mb:.1f}MB — maximum allowed is {MAX_PDF_SIZE_MB}MB"
    return True, ""
 
 
def build_pdf_index(file_bytes, filename):
    """
    Builds FAISS + BM25 index for a single PDF.
    Returns index dict with all needed components.
    """
    tmp_path = f"tmp_{filename.replace(' ', '_')}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
 
    try:
        docs       = load_pdf(tmp_path)
        split_docs = split_text(docs)
 
        # Enforce page limit
        if len(docs) > MAX_PAGES:
            docs       = docs[:MAX_PAGES]
            split_docs = split_text(docs)
 
        db          = create_db(split_docs)
        bm25, all_d = build_bm25_index(split_docs)
 
        return {
            "filename":   filename,
            "db":         db,
            "bm25":       bm25,
            "all_docs":   all_d,
            "split_docs": split_docs,
            "page_count": len(docs),
            "chunk_count": len(split_docs),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
 
 
# ───────────────────────────────────────────────
# TIER 3 — MULTI-PDF RETRIEVAL ← NEW
# Searches ALL uploaded PDFs simultaneously
# Tags each result with source PDF name
# ───────────────────────────────────────────────
def multi_pdf_retrieve(pdf_indexes, query, k_per_pdf=3):
    """
    Searches across all PDF indexes simultaneously.
    Returns merged results tagged with source PDF.
 
    Each returned doc has metadata:
    - source_pdf: filename of the PDF
    - page: page number
    """
    all_results = []
 
    for idx in pdf_indexes:
        db        = idx["db"]
        bm25      = idx["bm25"]
        all_docs  = idx["all_docs"]
        filename  = idx["filename"]
 
        try:
            # FAISS search
            faiss_results = db.similarity_search_with_score(query, k=k_per_pdf)
            faiss_docs    = [r[0] for r in faiss_results]
            faiss_scores  = [r[1] for r in faiss_results]
 
            # BM25 search
            if BM25_AVAILABLE and bm25 is not None:
                tokenized = query.lower().split()
                bm25_sc   = bm25.get_scores(tokenized)
                top_idx   = sorted(range(len(bm25_sc)),
                                   key=lambda i: bm25_sc[i], reverse=True)[:k_per_pdf]
                bm25_docs = [all_docs[i] for i in top_idx]
                combined  = reciprocal_rank_fusion(faiss_docs, bm25_docs)[:k_per_pdf]
            else:
                combined = faiss_docs
 
            # Tag each doc with source PDF
            for rank, doc in enumerate(combined):
                score = faiss_scores[rank] if rank < len(faiss_scores) else 1.0
                confidence = round((1 / (1 + score)) * 100, 1)
                doc.metadata["source_pdf"] = filename
                all_results.append({
                    "doc":        doc,
                    "confidence": confidence,
                    "source_pdf": filename,
                    "rank":       rank
                })
 
        except Exception as e:
            continue
 
    # Sort all results by confidence across PDFs
    all_results.sort(key=lambda x: x["confidence"], reverse=True)
 
    # Take top k overall
    top_results = all_results[:6]
    docs        = [r["doc"] for r in top_results]
    avg_conf    = round(sum(r["confidence"] for r in top_results) / len(top_results), 1) if top_results else 65.0
 
    # Build source map: PDF → pages
    source_map = {}
    for r in top_results:
        pdf_name = r["source_pdf"]
        page     = r["doc"].metadata.get("page", 0) + 1
        if pdf_name not in source_map:
            source_map[pdf_name] = []
        if page not in source_map[pdf_name]:
            source_map[pdf_name].append(page)
 
    return docs, avg_conf, source_map
 
 
# ───────────────────────────────────────────────
# SOURCE PAGES (single PDF)
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
        topic    = response.content.strip().strip('"').strip("'")
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
# GENERATE ANSWER (supports multi-PDF context)
# ───────────────────────────────────────────────
def generate_answer(query, docs, memory=None):
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source_pdf', 'Document')}]\n{d.page_content}"
        for d in docs
    ])
 
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
- If context comes from multiple documents, synthesize the information
 
{"Previous conversation:" + chr(10) + history_str + chr(10) if history_str else ""}
 
Context from PDF(s):
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
    context      = " ".join([d.page_content.lower() for d in docs])
    answer_lower = answer.lower()
    query_lower  = query.lower()
 
    stop_words = {"what","is","the","a","an","of","in","and","to","how",
                  "why","when","where","which","are","does","do","its",
                  "it","was","be","with","for"}
 
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
        data     = {"question": [query], "answer": [answer],
                    "contexts": [contexts], "ground_truth": [answer]}
        dataset  = Dataset.from_dict(data)
        result   = ragas_eval(dataset,
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
# FLASHCARD GENERATOR
# ───────────────────────────────────────────────
def generate_flashcards(query, answer, docs, num_cards=5):
    context = "\n\n".join([d.page_content for d in docs])
    prompt  = f"""You are an expert study coach creating flashcards.
Create {num_cards} flashcards based on the question and answer.
Return ONLY a valid JSON array:
[{{"front":"Short concept (max 15 words)","back":"Clear explanation (max 50 words)","topic":"Topic name"}}]
 
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
# DOCUMENT INTELLIGENCE
# ───────────────────────────────────────────────
def analyze_document(docs):
    sample      = docs[:20] if len(docs) > 20 else docs
    context     = "\n\n".join([d.page_content for d in sample])
    total_pages = max([d.metadata.get("page", 0) for d in docs]) + 1 if docs else 1
 
    prompt = f"""Analyze this study material.
Return ONLY a valid JSON object:
{{"summary":"2-3 sentence overview","key_topics":["T1","T2","T3","T4","T5"],"difficulty":"Beginner","estimated_hours":3,"total_concepts":12,"suggested_question":"Specific question"}}
difficulty must be: "Beginner", "Intermediate", or "Advanced"
Study Material: {context}"""
 
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
            "suggested_question": "What are the main topics?",
            "total_pages": total_pages,
            "total_chunks": len(docs)
        }
 
 
# ───────────────────────────────────────────────
# VOICE TRANSCRIPTION
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
                model="whisper-1", file=audio_file, language="en"
            )
        return transcript.text
    except Exception as e:
        return f"Transcription failed: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
 
 
# ───────────────────────────────────────────────
# ADAPTIVE MCQ GENERATION
# ───────────────────────────────────────────────
def generate_adaptive_mcqs(docs, num_questions=5, difficulty="Medium"):
    context = "\n\n".join([d.page_content for d in docs])
 
    difficulty_instructions = {
        "Easy":   "Ask simple recall questions. Clear language. Obvious distractors. Add a small hint.",
        "Medium": "Ask application questions. Plausible distractors. Test concept relationships.",
        "Hard":   "Ask analysis questions. Tricky distractors. Edge cases. Scenario-based."
    }
 
    prompt = f"""You are an expert exam creator making a {difficulty.upper()} difficulty quiz.
Guidelines: {difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])}
 
Generate {num_questions} MCQs. Return ONLY a valid JSON array:
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"...","difficulty":"{difficulty}","hint":"hint for Easy only else empty"}}]
 
Context: {context}"""
 
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
 
 
def generate_mcqs(docs, num_questions=5):
    return generate_adaptive_mcqs(docs, num_questions, "Medium")
 
 
# ───────────────────────────────────────────────
# CONCEPT MAP GENERATOR
# ───────────────────────────────────────────────
def generate_concept_map(docs):
    sample  = docs[:15] if len(docs) > 15 else docs
    context = "\n\n".join([d.page_content for d in sample])
 
    prompt = f"""You are an academic knowledge graph expert.
Analyze this material and identify key concepts and relationships.
 
Return ONLY a valid JSON object:
{{"nodes":[{{"id":"n1","label":"Machine Learning","category":"main"}}],"edges":[{{"source":"n1","target":"n2","relationship":"includes"}}]}}
 
Rules:
- 8-15 nodes total
- category: "main", "sub", or "detail"
- relationship: short verb like "includes", "uses", "requires", "leads to"
 
Study Material: {context}"""
 
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
# EXAM MODE
# ───────────────────────────────────────────────
def generate_exam(docs, num_mcq=10, num_short=3, difficulty="Medium"):
    context = "\n\n".join([d.page_content for d in docs[:15]])
 
    prompt = f"""You are an expert exam paper setter creating a {difficulty} difficulty exam.
Generate {num_mcq} MCQs and {num_short} short answer questions.
 
Return ONLY a valid JSON object:
{{"exam_title":"Mock Exam: [Subject]","difficulty":"{difficulty}","total_marks":{num_mcq + (num_short * 5)},"mcqs":[{{"id":1,"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","marks":1,"explanation":"..."}}],"short_answers":[{{"id":1,"question":"Explain...","marks":5,"key_points":["P1","P2","P3"],"model_answer":"Brief answer"}}]}}
 
Context: {context}"""
 
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
 
    weak_str = ""
    if weak_topics:
        top = sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_str = f"\nPrioritize weak topics: {[t[0] for t in top]}\n"
 
    prompt = f"""Create a personalized {num_days}-day study plan.{weak_str}
Return ONLY a valid JSON object:
{{"plan_title":"{num_days}-Day Study Plan: [Subject]","total_days":{num_days},"days":[{{"day":1,"title":"...","topics":["T1","T2"],"study_focus":"One sentence","suggested_questions":["Q1?","Q2?","Q3?"],"difficulty":"Easy","estimated_hours":2}}]}}
 
Study Material: {context}"""
 
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