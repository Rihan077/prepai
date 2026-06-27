import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from rag_pipeline import (
    load_pdf, split_text, create_db,
    build_bm25_index, hybrid_retrieve,
    retrieve_with_scores, get_sources,
    detect_topic, is_safe_query,
    generate_answer, generate_mcqs,
    evaluate_answer, create_memory,
    generate_study_plan,
    BM25_AVAILABLE, RAGAS_AVAILABLE
)

st.set_page_config(
    page_title="PrepAI — Smart Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
* { font-family: 'Segoe UI', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: #faf8f4;
}
[data-testid="stSidebar"] {
    background: #0a1628;
    border-right: 1px solid #1e3a6e;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }

h1, h2, h3 { color: #0f2040; }
p, li { color: #334155; }

.stTextInput > div > div > input {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    color: #0f2040;
    padding: 10px 14px;
}
.stTextInput > div > div > input:focus {
    border-color: #f59e0b;
    box-shadow: 0 0 0 3px #f59e0b20;
}

div[data-testid="stButton"] button {
    background: #0a1628;
    color: #f59e0b;
    border: 1.5px solid #f59e0b;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    padding: 8px 20px;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: #f59e0b;
    color: #0a1628;
}

.stSelectbox > div > div {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    color: #0f2040;
}

.metric-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-top: 3px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px #0f204008;
}
.metric-val {
    font-size: 26px;
    font-weight: 700;
    color: #0a1628;
}
.metric-label {
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.eval-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.eval-title {
    color: #0a1628;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.eval-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 8px 0;
}
.eval-metric { color: #64748b; font-size: 13px; }
.eval-score  { font-weight: 700; font-size: 14px; }
.score-high  { color: #059669; }
.score-mid   { color: #d97706; }
.score-low   { color: #dc2626; }
.eval-bar-bg { background: #f1f5f9; border-radius: 4px; height: 6px; margin-top: 4px; }
.eval-bar    { height: 6px; border-radius: 4px; }

.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
}
.badge-hybrid   { background: #fef3c7; color: #92400e; }
.badge-semantic { background: #f1f5f9; color: #64748b; }
.badge-ragas    { background: #d1fae5; color: #065f46; }
.badge-local    { background: #f1f5f9; color: #64748b; }

.chat-user {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #0f2040;
}
.chat-ai {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-left: 4px solid #f59e0b;
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #0f2040;
    box-shadow: 0 2px 8px #0f204008;
}
.user-label { color: #0a1628; font-size: 11px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px; }
.ai-label   { color: #d97706; font-size: 11px; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px; }

.quiz-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.quiz-q       { color: #0f2040; font-size: 15px; font-weight: 600; margin-bottom: 10px; }
.correct-ans  { color: #059669; font-weight: 700; }
.wrong-ans    { color: #dc2626; font-weight: 700; }
.explanation  { color: #64748b; font-size: 13px; margin-top: 8px; padding: 8px 12px; background: #fafafa; border-radius: 8px; }

.topic-badge {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px 3px 3px 0;
    border: 1px solid #fde68a;
}

.system-status {
    background: #0f2040;
    border: 1px solid #1e3a6e;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #94a3b8 !important;
}

.day-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-top: 3px solid #f59e0b;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.day-card-done {
    background: #f0fdf4;
    border: 1.5px solid #bbf7d0;
    border-top: 3px solid #059669;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}
.day-number   { font-size: 22px; font-weight: 800; color: #0a1628; }
.day-title    { font-size: 15px; font-weight: 600; color: #0f2040; margin-top: 4px; }
.day-topic-chip {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px;
    font-weight: 600;
}
.day-focus {
    color: #334155;
    font-size: 13px;
    margin: 10px 0;
    padding: 10px 14px;
    background: #faf8f4;
    border-radius: 8px;
    border-left: 3px solid #f59e0b;
}
.day-q {
    color: #475569;
    font-size: 13px;
    padding: 5px 0;
    border-bottom: 1px solid #f1f5f9;
}
.difficulty-easy   { color: #059669; font-weight: 700; font-size: 12px; }
.difficulty-medium { color: #d97706; font-weight: 700; font-size: 12px; }
.difficulty-hard   { color: #dc2626; font-weight: 700; font-size: 12px; }

.progress-bar-bg { background: #f1f5f9; border-radius: 8px; height: 10px; margin: 8px 0; }
.progress-bar    { height: 10px; border-radius: 8px; background: linear-gradient(90deg, #0a1628, #f59e0b); }

.prepai-header {
    background: #0a1628;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.section-divider {
    border: none;
    border-top: 2px solid #f1f5f9;
    margin: 20px 0;
}

.upload-area {
    background: #ffffff;
    border: 2px dashed #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}

.feature-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    box-shadow: 0 2px 8px #0f204008;
    transition: all 0.2s;
}
.feature-icon { font-size: 28px; margin-bottom: 10px; }
.feature-title { color: #0a1628; font-size: 15px; font-weight: 700; margin-bottom: 6px; }
.feature-desc  { color: #64748b; font-size: 13px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────
def init_state():
    defaults = {
        "db": None, "bm25_index": None, "all_docs": None,
        "chat": [], "memory": None, "topics": {},
        "quiz_data": None, "quiz_submitted": False,
        "user_answers": {}, "score": 0, "pdf_name": None,
        "study_plan": None, "completed_days": set(),
        "quiz_history": [], "question_log": [], "split_docs": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── SIDEBAR ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:32px'>🎓</div>
        <div style='font-size:22px;font-weight:800;color:#f59e0b;letter-spacing:1px'>PrepAI</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px'>Smart Study Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.selectbox(
        "Mode",
        ["💬 Study Chat", "🧪 Test Mode", "📅 Study Plan", "📈 Analytics"]
    )

    st.markdown("---")
    st.markdown('<div style="color:#f59e0b;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">SYSTEM STATUS</div>', unsafe_allow_html=True)

    bm25_status = "Active" if BM25_AVAILABLE else "Install rank-bm25"
    ragas_status = "Active" if RAGAS_AVAILABLE else "Local eval"
    bm25_color  = "#f59e0b" if BM25_AVAILABLE else "#dc2626"
    ragas_color = "#f59e0b" if RAGAS_AVAILABLE else "#94a3b8"

    st.markdown(
        f'<div class="system-status">🔍 Hybrid Search &nbsp;<b style="color:{bm25_color}">{bm25_status}</b></div>'
        f'<div class="system-status">📊 RAGAS Eval &nbsp;<b style="color:{ragas_color}">{ragas_status}</b></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown('<div style="color:#f59e0b;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">WEAK TOPICS</div>', unsafe_allow_html=True)

    if st.session_state.topics:
        for topic, count in sorted(
            st.session_state.topics.items(), key=lambda x: x[1], reverse=True
        ):
            st.markdown(f'<div class="topic-badge">📌 {topic} — {count}x</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:12px">Ask questions to track weak topics.</div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.db is not None:
        st.markdown(f'<div style="color:#f59e0b;font-size:12px;font-weight:600">📄 {st.session_state.pdf_name}</div>', unsafe_allow_html=True)
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:11px;text-align:center">GPT-4o-mini · FAISS · BM25 · LangChain</div>', unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────
st.markdown("""
<div class="prepai-header">
    <div>
        <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:1px">
            🎓 PrepAI
        </div>
        <div style="color:#94a3b8;font-size:14px;margin-top:4px">
            AI-powered adaptive learning — upload any PDF and start learning smarter
        </div>
    </div>
    <div style="text-align:right">
        <div style="color:#f59e0b;font-size:12px;font-weight:700">POWERED BY</div>
        <div style="color:#64748b;font-size:11px;margin-top:2px">RAG · BM25 · GPT-4o-mini</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── PDF UPLOAD ────────────────────────────────────
uploaded_file = st.file_uploader("Upload your study material (PDF)", type=["pdf"])


@st.cache_resource
def process_pdf(file_bytes, filename):
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)
    docs = load_pdf("temp.pdf")
    split_docs = split_text(docs)
    db = create_db(split_docs)
    bm25_index, all_docs = build_bm25_index(split_docs)
    return db, bm25_index, all_docs, split_docs


if uploaded_file:
    file_bytes = uploaded_file.read()
    filename   = uploaded_file.name

    if st.session_state.db is None or st.session_state.pdf_name != filename:
        with st.spinner("Building search indexes..."):
            db, bm25_index, all_docs, split_docs = process_pdf(file_bytes, filename)
            st.session_state.db         = db
            st.session_state.bm25_index = bm25_index
            st.session_state.all_docs   = all_docs
            st.session_state.split_docs = split_docs
            st.session_state.pdf_name   = filename
            st.session_state.memory     = create_memory()
            st.session_state.chat       = []

    db         = st.session_state.db
    bm25_index = st.session_state.bm25_index
    all_docs   = st.session_state.all_docs

    st.success(f"✅ {filename} is ready")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


    # ════════════════════════════════════════════
    # MODE 1 — STUDY CHAT
    # ════════════════════════════════════════════
    if mode == "💬 Study Chat":
        st.markdown("## 💬 Study Chat")

        query = st.text_input(
            "Ask a question",
            placeholder="e.g. Explain gradient descent with an example",
            label_visibility="collapsed"
        )

        if st.button("Get Answer →", use_container_width=True):
            if not query.strip():
                st.warning("Please type a question first.")
            elif not is_safe_query(query):
                st.error("Query blocked — contains unsafe content.")
            else:
                with st.spinner("Searching and generating answer..."):
                    docs, confidence, search_type = hybrid_retrieve(db, bm25_index, all_docs, query)
                    sources     = get_sources(docs)
                    answer      = generate_answer(query, docs, st.session_state.memory)
                    eval_scores = evaluate_answer(query, answer, docs)
                    topic       = detect_topic(query)

                    st.session_state.memory.add_user_message(query)
                    st.session_state.memory.add_ai_message(answer)
                    st.session_state.topics[topic] = st.session_state.topics.get(topic, 0) + 1
                    st.session_state.question_log.append({
                        "query": query, "topic": topic,
                        "confidence": confidence,
                        "time": datetime.now().strftime("%H:%M")
                    })
                    st.session_state.chat.append({"role": "user", "content": query})
                    st.session_state.chat.append({
                        "role": "ai", "content": answer,
                        "confidence": confidence, "sources": sources,
                        "topic": topic, "search_type": search_type,
                        "eval": eval_scores
                    })

        if st.session_state.chat:
            last_ai = next((m for m in reversed(st.session_state.chat) if m["role"] == "ai"), None)
            if last_ai:
                search_badge = (
                    '<span class="badge badge-hybrid">⚡ HYBRID</span>'
                    if last_ai.get("search_type") == "hybrid"
                    else '<span class="badge badge-semantic">SEMANTIC</span>'
                )
                eval_badge = (
                    '<span class="badge badge-ragas">RAGAS</span>'
                    if last_ai.get("eval", {}).get("method") == "ragas"
                    else '<span class="badge badge-local">LOCAL EVAL</span>'
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-val">{last_ai["confidence"]}%</div>'
                        f'<div class="metric-label">Confidence {search_badge}</div></div>',
                        unsafe_allow_html=True)
                with c2:
                    pages    = last_ai["sources"]
                    page_str = ", ".join(str(p) for p in pages) if pages else "N/A"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-val" style="font-size:18px">p. {page_str}</div>'
                        f'<div class="metric-label">Source Pages</div></div>',
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-val" style="font-size:16px">{last_ai["topic"]}</div>'
                        f'<div class="metric-label">Topic Detected</div></div>',
                        unsafe_allow_html=True)

                ev = last_ai.get("eval", {})
                if ev:
                    def sc(s): return "score-high" if s >= 70 else "score-mid" if s >= 40 else "score-low"
                    def bc(s): return "#059669" if s >= 70 else "#d97706" if s >= 40 else "#dc2626"
                    f_s  = ev.get("faithfulness", 0)
                    r_s  = ev.get("answer_relevancy", 0)
                    cp_s = ev.get("context_precision", 0)
                    m    = ev.get("method", "local")
                    st.markdown(
                        f'<div class="eval-card">'
                        f'<div class="eval-title">Answer Quality Evaluation {"(RAGAS)" if m=="ragas" else "(LOCAL)"} {eval_badge}</div>'
                        f'<div class="eval-row"><span class="eval-metric">Faithfulness</span><span class="eval-score {sc(f_s)}">{f_s}%</span></div>'
                        f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{f_s}%;background:{bc(f_s)}"></div></div>'
                        f'<div class="eval-row" style="margin-top:10px"><span class="eval-metric">Answer Relevancy</span><span class="eval-score {sc(r_s)}">{r_s}%</span></div>'
                        f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{r_s}%;background:{bc(r_s)}"></div></div>'
                        f'<div class="eval-row" style="margin-top:10px"><span class="eval-metric">Context Precision</span><span class="eval-score {sc(cp_s)}">{cp_s}%</span></div>'
                        f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{cp_s}%;background:{bc(cp_s)}"></div></div>'
                        f'</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="user-label">YOU</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai"><div class="ai-label">PREPAI</div>{msg["content"]}</div>', unsafe_allow_html=True)


    # ════════════════════════════════════════════
    # MODE 2 — TEST MODE
    # ════════════════════════════════════════════
    elif mode == "🧪 Test Mode":
        st.markdown("## 🧪 Test Mode")
        col1, col2 = st.columns([2, 1])
        with col1:
            topic_focus = st.text_input("Focus topic (optional)", placeholder="e.g. neural networks, clustering")
        with col2:
            num_q = st.selectbox("Questions", [3, 5, 7, 10], index=1)

        if st.button("Generate Quiz →", use_container_width=True):
            search_query = topic_focus.strip() if topic_focus.strip() else "important concepts key topics"
            with st.spinner("Generating quiz..."):
                docs, _, _ = retrieve_with_scores(db, search_query, k=5)
                questions  = generate_mcqs(docs, num_questions=num_q)
                if questions:
                    st.session_state.quiz_data      = questions
                    st.session_state.quiz_submitted  = False
                    st.session_state.user_answers    = {}
                    st.session_state.score           = 0
                else:
                    st.error("Failed to generate quiz. Try a larger PDF.")

        if st.session_state.quiz_data:
            questions = st.session_state.quiz_data
            st.markdown(f"### Quiz — {len(questions)} Questions")

            with st.form("quiz_form"):
                for i, q in enumerate(questions):
                    st.markdown(f'<div class="quiz-card"><div class="quiz-q">Q{i+1}. {q["question"]}</div></div>', unsafe_allow_html=True)
                    options     = q.get("options", {})
                    option_list = [f"{k}. {v}" for k, v in options.items()]
                    selected    = st.radio(f"Answer Q{i+1}", option_list, key=f"quiz_q_{i}", label_visibility="collapsed")
                    if selected:
                        st.session_state.user_answers[i] = selected[0]

                if st.form_submit_button("Submit Answers →", use_container_width=True):
                    score = sum(
                        1 for i, q in enumerate(questions)
                        if st.session_state.user_answers.get(i, "").upper() == q.get("answer", "").strip().upper()
                    )
                    st.session_state.score          = score
                    st.session_state.quiz_submitted = True
                    total = len(questions)
                    pct   = round((score / total) * 100)
                    st.session_state.quiz_history.append({
                        "score": score, "total": total, "pct": pct,
                        "topic": topic_focus.strip() if topic_focus.strip() else "General",
                        "time": datetime.now().strftime("%H:%M")
                    })

            if st.session_state.quiz_submitted:
                total = len(questions)
                score = st.session_state.score
                pct   = round((score / total) * 100)
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{score}/{total}</div><div class="metric-label">Score</div></div>', unsafe_allow_html=True)
                with r2:
                    col = "#059669" if pct >= 70 else "#d97706" if pct >= 40 else "#dc2626"
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{col}">{pct}%</div><div class="metric-label">Percentage</div></div>', unsafe_allow_html=True)
                with r3:
                    grade = "Excellent 🔥" if pct >= 80 else "Good 👍" if pct >= 60 else "Needs Work 📖"
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:16px">{grade}</div><div class="metric-label">Grade</div></div>', unsafe_allow_html=True)

                st.markdown("### Answer Review")
                for i, q in enumerate(questions):
                    correct      = q.get("answer", "").strip().upper()
                    user         = st.session_state.user_answers.get(i, "").strip().upper()
                    icon         = "✅" if user == correct else "❌"
                    correct_text = q.get("options", {}).get(correct, correct)
                    explanation  = q.get("explanation", "")
                    rc           = "correct-ans" if user == correct else "wrong-ans"
                    st.markdown(
                        f'<div class="quiz-card">'
                        f'<div class="quiz-q">{icon} Q{i+1}. {q["question"]}</div>'
                        f'<div class="{rc}">Your answer: {user} &nbsp;|&nbsp; Correct: {correct}. {correct_text}</div>'
                        f'{"<div class=\'explanation\'>💡 " + explanation + "</div>" if explanation else ""}'
                        f'</div>', unsafe_allow_html=True)


    # ════════════════════════════════════════════
    # MODE 3 — STUDY PLAN
    # ════════════════════════════════════════════
    elif mode == "📅 Study Plan":
        st.markdown("## 📅 Personalized Study Plan")
        st.markdown('<p style="color:#64748b">PrepAI analyzes your PDF and builds a day-by-day plan tailored to your weak areas.</p>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            num_days = st.selectbox("Plan Duration", [3, 5, 7, 10, 14], index=2)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.topics:
            weak_list = list(st.session_state.topics.keys())[:3]
            st.info(f"Detected weak topics from your chat: **{', '.join(weak_list)}** — these will be prioritized.")

        if st.button("Generate Study Plan →", use_container_width=True):
            with st.spinner("Analyzing PDF and building your plan..."):
                split_docs = st.session_state.get("split_docs", all_docs)
                plan = generate_study_plan(
                    split_docs, num_days=num_days,
                    weak_topics=st.session_state.topics if st.session_state.topics else None
                )
                if plan:
                    st.session_state.study_plan    = plan
                    st.session_state.completed_days = set()
                else:
                    st.error("Failed to generate plan. Try again.")

        if st.session_state.study_plan:
            plan      = st.session_state.study_plan
            days      = plan.get("days", [])
            completed = st.session_state.completed_days
            total_d   = len(days)
            done_c    = len(completed)
            pct       = round((done_c / total_d) * 100) if total_d else 0

            st.markdown(f"### {plan.get('plan_title', 'Study Plan')}")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="color:#64748b;font-size:13px">Progress</span>'
                f'<span style="color:#0a1628;font-size:13px;font-weight:700">{done_c}/{total_d} days — {pct}%</span>'
                f'</div>'
                f'<div class="progress-bar-bg"><div class="progress-bar" style="width:{pct}%"></div></div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            for day_data in days:
                day_num  = day_data.get("day", 0)
                is_done  = day_num in completed
                cc       = "day-card-done" if is_done else "day-card"
                diff     = day_data.get("difficulty", "Medium")
                dc       = f"difficulty-{diff.lower()}"
                di       = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff, "🟡")
                topics_html   = "".join(f'<span class="day-topic-chip">{t}</span>' for t in day_data.get("topics", []))
                questions_html = "".join(f'<div class="day-q">❓ {q}</div>' for q in day_data.get("suggested_questions", []))
                done_badge = ' <span style="color:#059669;font-size:13px;font-weight:700">✅ Completed</span>' if is_done else ""

                st.markdown(
                    f'<div class="{cc}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                    f'<span class="day-number">Day {day_num}{done_badge}</span>'
                    f'<span class="{dc}">{di} {diff} · ⏱ {day_data.get("estimated_hours", 2)}h</span>'
                    f'</div>'
                    f'<div class="day-title">{day_data.get("title", "")}</div>'
                    f'<div style="margin:10px 0">{topics_html}</div>'
                    f'<div class="day-focus">{day_data.get("study_focus", "")}</div>'
                    f'<div style="margin-top:10px"><div style="color:#94a3b8;font-size:11px;font-weight:700;margin-bottom:6px">SUGGESTED QUESTIONS</div>{questions_html}</div>'
                    f'</div>', unsafe_allow_html=True)

                lbl = "Mark Complete ✓" if not is_done else "Mark Incomplete"
                if st.button(lbl, key=f"day_btn_{day_num}"):
                    if is_done:
                        st.session_state.completed_days.discard(day_num)
                    else:
                        st.session_state.completed_days.add(day_num)
                    st.rerun()

            if done_c == total_d:
                st.balloons()
                st.success("🎉 Plan complete! You're ready for your exam!")


    # ════════════════════════════════════════════
    # MODE 4 — ANALYTICS
    # ════════════════════════════════════════════
    elif mode == "📈 Analytics":
        st.markdown("## 📈 Analytics Dashboard")

        has_data = bool(st.session_state.topics) or bool(st.session_state.quiz_history) or bool(st.session_state.question_log)

        if not has_data:
            st.info("No data yet — ask questions in Study Chat and take a quiz, then come back here.")
        else:
            q_count   = len(st.session_state.question_log)
            quiz_count = len(st.session_state.quiz_history)
            avg_score  = round(sum(q["pct"] for q in st.session_state.quiz_history) / quiz_count) if quiz_count else 0
            topic_count = len(st.session_state.topics)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{q_count}</div><div class="metric-label">Questions Asked</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{quiz_count}</div><div class="metric-label">Quizzes Taken</div></div>', unsafe_allow_html=True)
            with m3:
                col = "#059669" if avg_score >= 70 else "#d97706" if avg_score >= 40 else "#dc2626"
                st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{col}">{avg_score}%</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{topic_count}</div><div class="metric-label">Topics Explored</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cl, cr = st.columns(2)

            with cl:
                st.markdown("### Weak Topic Analysis")
                if st.session_state.topics:
                    sorted_t = sorted(st.session_state.topics.items(), key=lambda x: x[1], reverse=True)
                    t_names  = [t[0] for t in sorted_t]
                    t_counts = [t[1] for t in sorted_t]
                    colors   = ["#dc2626" if c >= 3 else "#d97706" if c >= 2 else "#0a1628" for c in t_counts]
                    fig = go.Figure(go.Bar(
                        x=t_counts, y=t_names, orientation='h',
                        marker_color=colors, text=t_counts, textposition='outside'
                    ))
                    fig.update_layout(
                        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                        font=dict(color='#334155'),
                        xaxis=dict(showgrid=False, color='#e2e8f0'),
                        yaxis=dict(showgrid=False, color='#334155'),
                        margin=dict(l=10, r=30, t=10, b=10), height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("🔴 Red = high weakness (3+ times) · 🟡 Yellow = moderate · 🔵 Navy = once")

            with cr:
                st.markdown("### Quiz Score History")
                if st.session_state.quiz_history:
                    quiz_nums  = [f"Quiz {i+1}" for i in range(len(st.session_state.quiz_history))]
                    quiz_pcts  = [q["pct"] for q in st.session_state.quiz_history]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=quiz_nums, y=quiz_pcts,
                        mode='lines+markers+text',
                        text=[f"{p}%" for p in quiz_pcts],
                        textposition='top center',
                        line=dict(color='#0a1628', width=2),
                        marker=dict(size=10, color=['#059669' if p >= 70 else '#d97706' if p >= 40 else '#dc2626' for p in quiz_pcts])
                    ))
                    fig2.add_hline(y=70, line_dash="dash", line_color="#f59e0b",
                                   opacity=0.6, annotation_text="Pass line (70%)")
                    fig2.update_layout(
                        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                        font=dict(color='#334155'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False, range=[0, 110]),
                        margin=dict(l=10, r=10, t=10, b=10), height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Take a quiz to see score history.")

            if st.session_state.question_log:
                st.markdown("### Recent Questions")
                for entry in reversed(st.session_state.question_log[-8:]):
                    conf_col = "#059669" if entry["confidence"] >= 70 else "#d97706"
                    st.markdown(
                        f'<div class="quiz-card" style="padding:12px 16px;margin:6px 0">'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="color:#0f2040;font-size:13px">❓ {entry["query"][:80]}{"..." if len(entry["query"]) > 80 else ""}</span>'
                        f'<span style="color:#94a3b8;font-size:11px">{entry["time"]}</span>'
                        f'</div>'
                        f'<div style="margin-top:6px">'
                        f'<span class="topic-badge">{entry["topic"]}</span>'
                        f'<span style="color:{conf_col};font-size:12px;margin-left:8px;font-weight:600">⚡ {entry["confidence"]}% confidence</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

            # ── EXPORT REPORT ────────────────────
            st.markdown("---")
            st.markdown("### 📥 Export Study Report")
            st.markdown('<p style="color:#64748b;font-size:13px">Download a summary of your session as a text report.</p>', unsafe_allow_html=True)

            if st.button("Download Session Report →", use_container_width=True):
                lines = []
                lines.append("=" * 60)
                lines.append("               PREPAI — STUDY SESSION REPORT")
                lines.append(f"               Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
                lines.append("=" * 60)
                lines.append("")

                lines.append("DOCUMENT STUDIED")
                lines.append("-" * 40)
                lines.append(f"  {st.session_state.pdf_name or 'N/A'}")
                lines.append("")

                lines.append("SESSION SUMMARY")
                lines.append("-" * 40)
                lines.append(f"  Questions Asked : {q_count}")
                lines.append(f"  Quizzes Taken   : {quiz_count}")
                lines.append(f"  Avg Quiz Score  : {avg_score}%")
                lines.append(f"  Topics Explored : {topic_count}")
                lines.append("")

                if st.session_state.topics:
                    lines.append("WEAK TOPICS DETECTED")
                    lines.append("-" * 40)
                    for t, c in sorted(st.session_state.topics.items(), key=lambda x: x[1], reverse=True):
                        level = "HIGH" if c >= 3 else "MEDIUM" if c >= 2 else "LOW"
                        lines.append(f"  [{level}] {t} — asked {c} time(s)")
                    lines.append("")

                if st.session_state.quiz_history:
                    lines.append("QUIZ HISTORY")
                    lines.append("-" * 40)
                    for i, qh in enumerate(st.session_state.quiz_history):
                        lines.append(f"  Quiz {i+1}: {qh['score']}/{qh['total']} ({qh['pct']}%) — {qh['topic']} at {qh['time']}")
                    lines.append("")

                if st.session_state.question_log:
                    lines.append("QUESTIONS ASKED")
                    lines.append("-" * 40)
                    for entry in st.session_state.question_log:
                        lines.append(f"  [{entry['time']}] {entry['query']}")
                        lines.append(f"           Topic: {entry['topic']} | Confidence: {entry['confidence']}%")
                    lines.append("")

                if st.session_state.study_plan:
                    lines.append("STUDY PLAN")
                    lines.append("-" * 40)
                    lines.append(f"  {st.session_state.study_plan.get('plan_title', '')}")
                    done = len(st.session_state.completed_days)
                    total_plan = len(st.session_state.study_plan.get("days", []))
                    lines.append(f"  Progress: {done}/{total_plan} days completed")
                    lines.append("")

                lines.append("=" * 60)
                lines.append("           Keep studying. PrepAI believes in you!")
                lines.append("=" * 60)

                report_text = "\n".join(lines)
                st.download_button(
                    label="Click here to download",
                    data=report_text,
                    file_name=f"PrepAI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ── NO PDF ────────────────────────────────────────
else:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("💬", "Study Chat", "Ask questions and get structured 2/5/10 mark answers from your material."),
        ("🧪", "Test Mode",  "Auto-generate MCQs with instant scoring, explanations, and review."),
        ("📅", "Study Plan", "Get a personalized day-by-day plan built directly from your PDF."),
        ("📈", "Analytics",  "Track weak topics, quiz scores, confidence trends, and export reports."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f'</div>', unsafe_allow_html=True)