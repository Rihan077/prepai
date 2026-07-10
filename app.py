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
    generate_flashcards, analyze_document, transcribe_audio,
    BM25_AVAILABLE, RAGAS_AVAILABLE
)
from auth import register_user, login_user
from database import (
    init_db, save_topic, get_topics,
    save_quiz, get_quiz_history, get_adaptive_difficulty,
    save_question, get_question_log,
    update_streak, get_streak, get_total_study_days,
    save_study_plan, get_latest_study_plan,
    update_completed_days, get_adaptive_k
)
 
init_db()
 
st.set_page_config(
    page_title="PrepAI — Smart Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
* { font-family: 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #faf8f4; }
[data-testid="stSidebar"] { background: #0a1628; border-right: 1px solid #1e3a6e; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
 
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
.metric-val { font-size: 26px; font-weight: 700; color: #0a1628; }
.metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
 
.eval-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.eval-title { color: #0a1628; font-size: 12px; font-weight: 700; margin-bottom: 14px; letter-spacing: 1px; text-transform: uppercase; }
.eval-row { display: flex; justify-content: space-between; align-items: center; margin: 8px 0; }
.eval-metric { color: #64748b; font-size: 13px; }
.eval-score { font-weight: 700; font-size: 14px; }
.score-high { color: #059669; }
.score-mid  { color: #d97706; }
.score-low  { color: #dc2626; }
.eval-bar-bg { background: #f1f5f9; border-radius: 4px; height: 6px; margin-top: 4px; }
.eval-bar { height: 6px; border-radius: 4px; }
 
.badge { display: inline-block; border-radius: 20px; padding: 2px 10px; font-size: 11px; font-weight: 600; margin-left: 8px; }
.badge-hybrid { background: #fef3c7; color: #92400e; }
.badge-semantic { background: #f1f5f9; color: #64748b; }
.badge-ragas { background: #d1fae5; color: #065f46; }
.badge-local { background: #f1f5f9; color: #64748b; }
.badge-easy { background: #d1fae5; color: #065f46; }
.badge-medium { background: #fef3c7; color: #92400e; }
.badge-hard { background: #fee2e2; color: #991b1b; }
 
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
.user-label { color: #0a1628; font-size: 11px; font-weight: 700; margin-bottom: 6px; }
.ai-label   { color: #d97706; font-size: 11px; font-weight: 700; margin-bottom: 6px; }
 
.quiz-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.quiz-q { color: #0f2040; font-size: 15px; font-weight: 600; margin-bottom: 10px; }
.correct-ans { color: #059669; font-weight: 700; }
.wrong-ans   { color: #dc2626; font-weight: 700; }
.explanation { color: #64748b; font-size: 13px; margin-top: 8px; padding: 8px 12px; background: #fafafa; border-radius: 8px; }
 
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
.day-number { font-size: 22px; font-weight: 800; color: #0a1628; }
.day-title  { font-size: 15px; font-weight: 600; color: #0f2040; margin-top: 4px; }
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
.day-q { color: #475569; font-size: 13px; padding: 5px 0; border-bottom: 1px solid #f1f5f9; }
.difficulty-easy   { color: #059669; font-weight: 700; font-size: 12px; }
.difficulty-medium { color: #d97706; font-weight: 700; font-size: 12px; }
.difficulty-hard   { color: #dc2626; font-weight: 700; font-size: 12px; }
.progress-bar-bg { background: #f1f5f9; border-radius: 8px; height: 10px; margin: 8px 0; }
.progress-bar { height: 10px; border-radius: 8px; background: linear-gradient(90deg, #0a1628, #f59e0b); }
 
.prepai-header {
    background: #0a1628;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.streak-card {
    background: #0f2040;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    text-align: center;
}
.feature-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px #0f204008;
}
.feature-icon { font-size: 28px; margin-bottom: 10px; }
.feature-title { color: #0a1628; font-size: 15px; font-weight: 700; margin-bottom: 6px; }
.feature-desc  { color: #64748b; font-size: 13px; line-height: 1.5; }
 
/* ── DOCUMENT INTELLIGENCE ── */
.doc-intel-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-left: 4px solid #0a1628;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    box-shadow: 0 2px 8px #0f204008;
}
.doc-intel-title { color: #0a1628; font-size: 13px; font-weight: 700; margin-bottom: 12px; letter-spacing: 1px; text-transform: uppercase; }
.doc-stat { display: inline-block; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 14px; margin: 4px; text-align: center; }
.doc-stat-val { font-size: 18px; font-weight: 700; color: #0a1628; }
.doc-stat-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; }
.topic-chip { display: inline-block; background: #0a1628; color: #f59e0b; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 600; margin: 3px; }
.diff-beginner { color: #059669; font-weight: 700; }
.diff-intermediate { color: #d97706; font-weight: 700; }
.diff-advanced { color: #dc2626; font-weight: 700; }
 
/* ── FLASHCARDS ── */
.flashcard-front {
    background: #0a1628;
    border-radius: 14px 14px 0 0;
    padding: 20px;
    text-align: center;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.flashcard-back {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 16px 20px;
    min-height: 80px;
}
.flashcard-front-text { color: #f59e0b; font-size: 15px; font-weight: 600; }
.flashcard-back-text { color: #334155; font-size: 14px; line-height: 1.6; }
.flashcard-topic { color: #94a3b8; font-size: 11px; margin-top: 8px; text-align: right; }
.fc-got-it { background: #d1fae5; border: 1.5px solid #6ee7b7; border-radius: 8px; padding: 6px 14px; color: #065f46; font-weight: 700; font-size: 12px; text-align: center; }
.fc-review { background: #fee2e2; border: 1.5px solid #fca5a5; border-radius: 8px; padding: 6px 14px; color: #991b1b; font-weight: 700; font-size: 12px; text-align: center; }
 
/* ── VOICE ── */
.voice-section {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
 
div[data-testid="stButton"] button {
    background: #0a1628;
    color: #f59e0b;
    border: 1.5px solid #f59e0b;
    border-radius: 10px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover {
    background: #f59e0b;
    color: #0a1628;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── SESSION STATE ─────────────────────────────────
def init_state():
    defaults = {
        "user": None,
        "db": None, "bm25_index": None, "all_docs": None,
        "chat": [], "memory": None,
        "quiz_data": None, "quiz_submitted": False,
        "user_answers": {}, "score": 0, "pdf_name": None,
        "current_plan_id": None, "split_docs": None,
        "completed_days": set(), "study_plan": None,
        "current_difficulty": "Medium",
        "doc_intel": None,
        "flashcards": [],
        "flashcard_status": {},
        "current_fc_index": 0,
        "voice_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
 
init_state()
 
 
# ════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════
def show_auth_page():
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px">
        <div style="font-size:48px">🎓</div>
        <div style="font-size:36px;font-weight:800;color:#0a1628;letter-spacing:1px">PrepAI</div>
        <div style="font-size:14px;color:#64748b;margin-top:4px">AI-powered adaptive learning</div>
    </div>
    """, unsafe_allow_html=True)
 
    col = st.columns([1, 2, 1])[1]
    with col:
        tab1, tab2 = st.tabs(["Login", "Create Account"])
 
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", key="login_pass", placeholder="Enter your password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login →", use_container_width=True, key="login_btn"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, user, msg = login_user(username, password)
                    if success:
                        st.session_state.user = user
                        update_streak(user["id"])
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error(msg)
 
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_username = st.text_input("Choose username", key="reg_user", placeholder="At least 3 characters")
            new_password = st.text_input("Choose password", key="reg_pass", placeholder="At least 6 characters", type="password")
            confirm_pass = st.text_input("Confirm password", key="reg_confirm", placeholder="Repeat your password", type="password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, key="reg_btn"):
                if not new_username or not new_password or not confirm_pass:
                    st.error("Please fill in all fields.")
                elif new_password != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(new_username, new_password)
                    if success:
                        st.success(msg + " Please login.")
                    else:
                        st.error(msg)
 
 
# ════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════
def show_main_app():
    user    = st.session_state.user
    user_id = user["id"]
 
    topics       = get_topics(user_id)
    quiz_history = get_quiz_history(user_id)
    question_log = get_question_log(user_id)
    streak       = get_streak(user_id)
    total_days   = get_total_study_days(user_id)
 
    # ── SIDEBAR ───────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:16px 0 8px'>
            <div style='font-size:32px'>🎓</div>
            <div style='font-size:22px;font-weight:800;color:#f59e0b;letter-spacing:1px'>PrepAI</div>
            <div style='font-size:11px;color:#64748b;margin-top:2px'>Smart Study Assistant</div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown(
            f'<div style="background:#0f2040;border-radius:10px;padding:12px;margin:8px 0;text-align:center">'
            f'<div style="color:#f59e0b;font-size:13px;font-weight:700">👤 {user["username"]}</div>'
            f'<div style="color:#475569;font-size:11px;margin-top:2px">Member since {user["created_at"][:10]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="streak-card">'
            f'<div style="color:#f59e0b;font-size:22px;font-weight:800">{streak} 🔥</div>'
            f'<div style="color:#94a3b8;font-size:11px;margin-top:2px">Day streak</div>'
            f'<div style="color:#64748b;font-size:11px;margin-top:4px">{total_days} total study days</div>'
            f'</div>',
            unsafe_allow_html=True
        )
 
        st.markdown("---")
        mode = st.selectbox("Mode", [
            "💬 Study Chat",
            "🧪 Test Mode",
            "📅 Study Plan",
            "📈 Analytics",
            "🃏 Flashcards"
        ])
 
        st.markdown("---")
        st.markdown('<div style="color:#f59e0b;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">SYSTEM STATUS</div>', unsafe_allow_html=True)
        bm25_color  = "#f59e0b" if BM25_AVAILABLE else "#dc2626"
        ragas_color = "#f59e0b" if RAGAS_AVAILABLE else "#94a3b8"
        st.markdown(
            f'<div class="system-status">🔍 Hybrid Search &nbsp;<b style="color:{bm25_color}">{"Active" if BM25_AVAILABLE else "Install rank-bm25"}</b></div>'
            f'<div class="system-status">📊 RAGAS Eval &nbsp;<b style="color:{ragas_color}">{"Active" if RAGAS_AVAILABLE else "Local eval"}</b></div>'
            f'<div class="system-status">🎙️ Voice Input &nbsp;<b style="color:#f59e0b">Active</b></div>',
            unsafe_allow_html=True
        )
 
        st.markdown("---")
        st.markdown('<div style="color:#f59e0b;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:8px">WEAK TOPICS</div>', unsafe_allow_html=True)
        if topics:
            for topic, count in list(topics.items())[:5]:
                st.markdown(f'<div class="topic-badge">📌 {topic} — {count}x</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:12px">Ask questions to track weak topics.</div>', unsafe_allow_html=True)
 
        st.markdown("---")
        if st.session_state.db is not None:
            st.markdown(f'<div style="color:#f59e0b;font-size:12px;font-weight:600">📄 {st.session_state.pdf_name}</div>', unsafe_allow_html=True)
 
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
 
        st.markdown("---")
        st.markdown('<div style="color:#475569;font-size:11px;text-align:center">GPT-4o-mini · FAISS · BM25 · Whisper</div>', unsafe_allow_html=True)
 
    # ── HEADER ────────────────────────────────────
    st.markdown(f"""
    <div class="prepai-header">
        <div>
            <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:1px">🎓 PrepAI</div>
            <div style="color:#94a3b8;font-size:14px;margin-top:4px">Welcome back, {user['username']} — let's study smarter today</div>
        </div>
        <div style="text-align:right">
            <div style="color:#f59e0b;font-size:12px;font-weight:700">POWERED BY</div>
            <div style="color:#64748b;font-size:11px;margin-top:2px">RAG · BM25 · GPT-4o-mini · Whisper</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── PDF UPLOAD ────────────────────────────────
    uploaded_file = st.file_uploader("Upload your study material (PDF)", type=["pdf"])
 
    @st.cache_resource
    def process_pdf(file_bytes, filename):
        with open("temp.pdf", "wb") as f:
            f.write(file_bytes)
        docs       = load_pdf("temp.pdf")
        split_docs = split_text(docs)
        db         = create_db(split_docs)
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
                st.session_state.doc_intel  = None
                st.session_state.flashcards = []
 
        db         = st.session_state.db
        bm25_index = st.session_state.bm25_index
        all_docs   = st.session_state.all_docs
        adaptive_k = get_adaptive_k(user_id)
 
        st.success(f"✅ {filename} is ready")
 
        # ── TIER 1: DOCUMENT INTELLIGENCE ─────────
        if st.session_state.doc_intel is None:
            with st.spinner("🔍 Analyzing document..."):
                st.session_state.doc_intel = analyze_document(
                    st.session_state.split_docs
                )
 
        intel = st.session_state.doc_intel
        if intel:
            diff_class = {
                "Beginner": "diff-beginner",
                "Intermediate": "diff-intermediate",
                "Advanced": "diff-advanced"
            }.get(intel.get("difficulty", "Intermediate"), "diff-intermediate")
 
            topics_html = "".join(
                f'<span class="topic-chip">{t}</span>'
                for t in intel.get("key_topics", [])
            )
 
            st.markdown(
                f'<div class="doc-intel-card">'
                f'<div class="doc-intel-title">📊 Document Intelligence</div>'
                f'<p style="color:#334155;font-size:14px;margin-bottom:14px">{intel.get("summary", "")}</p>'
                f'<div style="margin-bottom:12px">{topics_html}</div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">'
                f'<div class="doc-stat"><div class="doc-stat-val">{intel.get("total_pages", "?")}</div><div class="doc-stat-label">Pages</div></div>'
                f'<div class="doc-stat"><div class="doc-stat-val">{intel.get("total_concepts", "?")} </div><div class="doc-stat-label">Concepts</div></div>'
                f'<div class="doc-stat"><div class="doc-stat-val">{intel.get("estimated_hours", "?")}h</div><div class="doc-stat-label">Study Time</div></div>'
                f'<div class="doc-stat"><div class="doc-stat-val {diff_class}">{intel.get("difficulty", "?")}</div><div class="doc-stat-label">Difficulty</div></div>'
                f'</div>'
                f'<div style="background:#faf8f4;border-radius:8px;padding:10px 14px;border-left:3px solid #f59e0b">'
                f'<span style="color:#64748b;font-size:12px;font-weight:700">💡 SUGGESTED FIRST QUESTION: </span>'
                f'<span style="color:#334155;font-size:13px">{intel.get("suggested_question", "")}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
 
        st.markdown("---")
 
        # ════════════════════════════════════════
        # MODE 1 — STUDY CHAT
        # ════════════════════════════════════════
        if mode == "💬 Study Chat":
            st.markdown("## 💬 Study Chat")
 
            # ── TIER 1: VOICE INPUT ───────────────
            with st.expander("🎙️ Voice Input — speak your question"):
                st.markdown('<div class="voice-section">', unsafe_allow_html=True)
                st.caption("Click the microphone, speak your question, then click Get Answer.")
                audio_data = st.audio_input("Record your question", key="voice_input")
                if audio_data is not None:
                    with st.spinner("Transcribing..."):
                        audio_bytes = audio_data.read()
                        transcribed = transcribe_audio(audio_bytes)
                        if transcribed and not transcribed.startswith("Transcription failed"):
                            st.session_state.voice_query = transcribed
                            st.success(f"Transcribed: *{transcribed}*")
                        else:
                            st.error(transcribed)
                st.markdown('</div>', unsafe_allow_html=True)
 
            # Use voice query if available
            default_query = st.session_state.get("voice_query", "")
            query = st.text_input(
                "Ask a question",
                value=default_query,
                placeholder="e.g. Explain gradient descent with an example",
                label_visibility="collapsed",
                key="text_query"
            )
 
            if st.button("Get Answer →", use_container_width=True):
                # Clear voice query after use
                st.session_state.voice_query = ""
 
                if not query.strip():
                    st.warning("Please type or speak a question first.")
                elif not is_safe_query(query):
                    st.error("Query blocked — contains unsafe content.")
                else:
                    with st.spinner("Searching and generating answer..."):
                        docs, confidence, search_type = hybrid_retrieve(
                            db, bm25_index, all_docs, query, k=adaptive_k
                        )
                        sources     = get_sources(docs)
                        answer      = generate_answer(query, docs, st.session_state.memory)
                        eval_scores = evaluate_answer(query, answer, docs)
                        topic       = detect_topic(query)
 
                        st.session_state.memory.add_user_message(query)
                        st.session_state.memory.add_ai_message(answer)
 
                        save_topic(user_id, topic)
                        save_question(user_id, query, topic, float(confidence))
                        update_streak(user_id)
 
                        # Auto-generate flashcards
                        with st.spinner("Generating flashcards..."):
                            new_cards = generate_flashcards(query, answer, docs)
                            if new_cards:
                                existing = st.session_state.flashcards
                                existing.extend(new_cards)
                                st.session_state.flashcards = existing[-20:]
 
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
                            f'<div class="metric-val">{round(float(last_ai["confidence"]), 1)}%</div>'
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
                            f'<div class="eval-title">Answer Quality {"(RAGAS)" if m=="ragas" else "(LOCAL)"} {eval_badge}</div>'
                            f'<div class="eval-row"><span class="eval-metric">Faithfulness</span><span class="eval-score {sc(f_s)}">{f_s}%</span></div>'
                            f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{f_s}%;background:{bc(f_s)}"></div></div>'
                            f'<div class="eval-row" style="margin-top:10px"><span class="eval-metric">Answer Relevancy</span><span class="eval-score {sc(r_s)}">{r_s}%</span></div>'
                            f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{r_s}%;background:{bc(r_s)}"></div></div>'
                            f'<div class="eval-row" style="margin-top:10px"><span class="eval-metric">Context Precision</span><span class="eval-score {sc(cp_s)}">{cp_s}%</span></div>'
                            f'<div class="eval-bar-bg"><div class="eval-bar" style="width:{cp_s}%;background:{bc(cp_s)}"></div></div>'
                            f'</div>', unsafe_allow_html=True)
 
                    if st.session_state.flashcards:
                        st.info(f"🃏 {len(st.session_state.flashcards)} flashcards generated — go to **Flashcards** mode to review them!")
 
                    st.markdown("<br>", unsafe_allow_html=True)
 
            for msg in st.session_state.chat:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user"><div class="user-label">YOU</div>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai"><div class="ai-label">PREPAI</div>{msg["content"]}</div>', unsafe_allow_html=True)
 
        # ════════════════════════════════════════
        # MODE 2 — TEST MODE
        # ════════════════════════════════════════
        elif mode == "🧪 Test Mode":
            st.markdown("## 🧪 Test Mode")
            col1, col2 = st.columns([2, 1])
            with col1:
                topic_focus = st.text_input("Focus topic (optional)", placeholder="e.g. neural networks, clustering")
            with col2:
                num_q = st.selectbox("Questions", [3, 5, 7, 10], index=1)
 
            if topic_focus.strip():
                diff = get_adaptive_difficulty(user_id, topic_focus.strip())
                diff_colors = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}
                st.markdown(
                    f'<div style="margin-bottom:8px">Adaptive difficulty for <b>{topic_focus}</b>: '
                    f'<span class="badge {diff_colors[diff]}">{diff}</span> '
                    f'<span style="color:#64748b;font-size:12px">(based on your past performance)</span></div>',
                    unsafe_allow_html=True
                )
 
            if st.button("Generate Quiz →", use_container_width=True):
                search_query = topic_focus.strip() if topic_focus.strip() else "important concepts"
                difficulty   = get_adaptive_difficulty(user_id, search_query)
                with st.spinner(f"Generating {difficulty} quiz..."):
                    docs, _, _ = retrieve_with_scores(db, search_query, k=5)
                    questions  = generate_mcqs(docs, num_questions=num_q)
                    if questions:
                        st.session_state.quiz_data         = questions
                        st.session_state.quiz_submitted     = False
                        st.session_state.user_answers       = {}
                        st.session_state.score              = 0
                        st.session_state.current_difficulty = difficulty
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
                        total      = len(questions)
                        pct        = round((score / total) * 100)
                        difficulty = st.session_state.get("current_difficulty", "Medium")
                        st.session_state.score          = score
                        st.session_state.quiz_submitted = True
                        save_quiz(user_id,
                                  topic_focus.strip() if topic_focus.strip() else "General",
                                  score, total, pct, difficulty)
                        update_streak(user_id)
 
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
 
                    next_diff   = get_adaptive_difficulty(user_id, topic_focus.strip() if topic_focus.strip() else "General")
                    diff_colors = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}
                    st.markdown(
                        f'<div style="margin:12px 0">Next quiz difficulty: <span class="badge {diff_colors[next_diff]}">{next_diff}</span></div>',
                        unsafe_allow_html=True
                    )
 
                    st.markdown("### Answer Review")
                    for i, q in enumerate(questions):
                        correct      = q.get("answer", "").strip().upper()
                        user_ans     = st.session_state.user_answers.get(i, "").strip().upper()
                        icon         = "✅" if user_ans == correct else "❌"
                        correct_text = q.get("options", {}).get(correct, correct)
                        explanation  = q.get("explanation", "")
                        rc           = "correct-ans" if user_ans == correct else "wrong-ans"
                        st.markdown(
                            f'<div class="quiz-card">'
                            f'<div class="quiz-q">{icon} Q{i+1}. {q["question"]}</div>'
                            f'<div class="{rc}">Your answer: {user_ans} &nbsp;|&nbsp; Correct: {correct}. {correct_text}</div>'
                            f'{"<div class=\'explanation\'>💡 " + explanation + "</div>" if explanation else ""}'
                            f'</div>', unsafe_allow_html=True)
 
        # ════════════════════════════════════════
        # MODE 3 — STUDY PLAN
        # ════════════════════════════════════════
        elif mode == "📅 Study Plan":
            st.markdown("## 📅 Personalized Study Plan")
            st.markdown('<p style="color:#64748b">PrepAI analyzes your PDF and builds a day-by-day plan tailored to your weak areas.</p>', unsafe_allow_html=True)
 
            existing_plan = get_latest_study_plan(user_id)
            if existing_plan and not st.session_state.get("study_plan"):
                st.session_state.study_plan      = existing_plan["plan_data"]
                st.session_state.current_plan_id  = existing_plan["id"]
                st.session_state.completed_days   = existing_plan["completed_days"]
 
            col1, col2 = st.columns([2, 1])
            with col1:
                num_days = st.selectbox("Plan Duration", [3, 5, 7, 10, 14], index=2)
 
            if topics:
                weak_list = list(topics.keys())[:3]
                st.info(f"Prioritizing your weak areas: **{', '.join(weak_list)}**")
 
            if st.button("Generate New Study Plan →", use_container_width=True):
                with st.spinner("Building your personalized plan..."):
                    split_docs = st.session_state.get("split_docs", all_docs)
                    plan = generate_study_plan(
                        split_docs, num_days=num_days,
                        weak_topics=topics if topics else None
                    )
                    if plan:
                        plan_id = save_study_plan(user_id, plan.get("plan_title", "Study Plan"), plan)
                        st.session_state.study_plan      = plan
                        st.session_state.current_plan_id  = plan_id
                        st.session_state.completed_days   = set()
                    else:
                        st.error("Failed to generate plan. Try again.")
 
            if st.session_state.get("study_plan"):
                plan      = st.session_state.study_plan
                days      = plan.get("days", [])
                completed = st.session_state.get("completed_days", set())
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
                    day_num        = day_data.get("day", 0)
                    is_done        = day_num in completed
                    cc             = "day-card-done" if is_done else "day-card"
                    diff           = day_data.get("difficulty", "Medium")
                    dc             = f"difficulty-{diff.lower()}"
                    di             = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff, "🟡")
                    topics_html    = "".join(f'<span class="day-topic-chip">{t}</span>' for t in day_data.get("topics", []))
                    questions_html = "".join(f'<div class="day-q">❓ {q}</div>' for q in day_data.get("suggested_questions", []))
                    done_badge     = ' <span style="color:#059669;font-size:13px;font-weight:700">✅ Completed</span>' if is_done else ""
 
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
                            completed.discard(day_num)
                        else:
                            completed.add(day_num)
                        st.session_state.completed_days = completed
                        if st.session_state.current_plan_id:
                            update_completed_days(st.session_state.current_plan_id, completed)
                        st.rerun()
 
                if done_c == total_d:
                    st.balloons()
                    st.success("🎉 Plan complete! You are ready for your exam!")
 
        # ════════════════════════════════════════
        # MODE 4 — ANALYTICS
        # ════════════════════════════════════════
        elif mode == "📈 Analytics":
            st.markdown("## 📈 Analytics Dashboard")
 
            has_data = bool(topics) or bool(quiz_history) or bool(question_log)
 
            if not has_data:
                st.info("No data yet — ask questions and take a quiz first.")
            else:
                q_count     = len(question_log)
                quiz_count  = len(quiz_history)
                avg_score   = round(sum(q["pct"] for q in quiz_history) / quiz_count) if quiz_count else 0
                topic_count = len(topics)
                fc_count    = len(st.session_state.flashcards)
 
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{q_count}</div><div class="metric-label">Questions Asked</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{quiz_count}</div><div class="metric-label">Quizzes Taken</div></div>', unsafe_allow_html=True)
                with m3:
                    col = "#059669" if avg_score >= 70 else "#d97706" if avg_score >= 40 else "#dc2626"
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{col}">{avg_score}%</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{streak} 🔥</div><div class="metric-label">Day Streak</div></div>', unsafe_allow_html=True)
                with m5:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{fc_count}</div><div class="metric-label">Flashcards</div></div>', unsafe_allow_html=True)
 
                st.markdown("<br>", unsafe_allow_html=True)
                cl, cr = st.columns(2)
 
                with cl:
                    st.markdown("### Weak Topic Analysis")
                    if topics:
                        sorted_t = sorted(topics.items(), key=lambda x: x[1], reverse=True)
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
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False),
                            margin=dict(l=10, r=30, t=10, b=10), height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🔴 Red = high weakness · 🟡 Yellow = moderate · 🔵 Navy = once")
 
                with cr:
                    st.markdown("### Quiz Score History")
                    if quiz_history:
                        quiz_nums      = [f"Quiz {i+1}" for i in range(len(quiz_history))]
                        quiz_pcts      = [q["pct"] for q in quiz_history]
                        diff_color_map = {"Easy": "#059669", "Medium": "#d97706", "Hard": "#dc2626"}
                        marker_colors  = [diff_color_map.get(q.get("difficulty", "Medium"), "#0a1628") for q in quiz_history]
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=quiz_nums, y=quiz_pcts,
                            mode='lines+markers+text',
                            text=[f"{p}%" for p in quiz_pcts],
                            textposition='top center',
                            line=dict(color='#0a1628', width=2),
                            marker=dict(size=10, color=marker_colors)
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
                        st.caption("🟢 Easy · 🟡 Medium · 🔴 Hard quiz")
                    else:
                        st.info("Take a quiz to see score history.")
 
                if question_log:
                    st.markdown("### Recent Questions")
                    for entry in question_log[:8]:
                        conf_col = "#059669" if float(entry["confidence"]) >= 70 else "#d97706"
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
 
                st.markdown("---")
                st.markdown("### 📥 Export Study Report")
                if st.button("Download Session Report →", use_container_width=True):
                    lines = []
                    lines.append("=" * 60)
                    lines.append("          PREPAI — STUDY SESSION REPORT")
                    lines.append(f"          User: {user['username']}")
                    lines.append(f"          Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
                    lines.append("=" * 60)
                    lines.append(f"\nSTREAK: {streak} days 🔥 | Total study days: {total_days}\n")
                    lines.append("SESSION SUMMARY")
                    lines.append("-" * 40)
                    lines.append(f"  Questions Asked : {q_count}")
                    lines.append(f"  Quizzes Taken   : {quiz_count}")
                    lines.append(f"  Avg Quiz Score  : {avg_score}%")
                    lines.append(f"  Topics Explored : {topic_count}")
                    lines.append(f"  Flashcards Made : {fc_count}\n")
 
                    if topics:
                        lines.append("WEAK TOPICS")
                        lines.append("-" * 40)
                        for t, c in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                            level = "HIGH" if c >= 3 else "MEDIUM" if c >= 2 else "LOW"
                            lines.append(f"  [{level}] {t} — asked {c} time(s)")
                        lines.append("")
 
                    if quiz_history:
                        lines.append("QUIZ HISTORY")
                        lines.append("-" * 40)
                        for i, qh in enumerate(quiz_history):
                            lines.append(f"  Quiz {i+1}: {qh['score']}/{qh['total']} ({qh['pct']}%) [{qh.get('difficulty','Medium')}] — {qh['topic']}")
                        lines.append("")
 
                    lines.append("=" * 60)
                    lines.append("     Keep studying. PrepAI believes in you!")
                    lines.append("=" * 60)
 
                    report_text = "\n".join(lines)
                    st.download_button(
                        label="Click to download",
                        data=report_text,
                        file_name=f"PrepAI_Report_{user['username']}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
 
        # ════════════════════════════════════════
        # MODE 5 — FLASHCARDS ← NEW
        # ════════════════════════════════════════
        elif mode == "🃏 Flashcards":
            st.markdown("## 🃏 Flashcard Review")
            st.caption("Flashcards are auto-generated every time you ask a question in Study Chat.")
 
            flashcards = st.session_state.flashcards
 
            if not flashcards:
                st.info("No flashcards yet — ask a question in Study Chat and flashcards will be generated automatically!")
            else:
                total_fc  = len(flashcards)
                got_it    = sum(1 for i, s in st.session_state.flashcard_status.items() if s == "got_it")
                review    = sum(1 for i, s in st.session_state.flashcard_status.items() if s == "review")
                remaining = total_fc - len(st.session_state.flashcard_status)
 
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_fc}</div><div class="metric-label">Total Cards</div></div>', unsafe_allow_html=True)
                with fc2:
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#059669">{got_it}</div><div class="metric-label">Got It ✓</div></div>', unsafe_allow_html=True)
                with fc3:
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#dc2626">{review}</div><div class="metric-label">Review Again</div></div>', unsafe_allow_html=True)
                with fc4:
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#d97706">{remaining}</div><div class="metric-label">Remaining</div></div>', unsafe_allow_html=True)
 
                st.markdown("<br>", unsafe_allow_html=True)
 
                # Navigation
                idx = st.session_state.current_fc_index
                if idx >= total_fc:
                    idx = 0
                    st.session_state.current_fc_index = 0
 
                card = flashcards[idx]
 
                st.markdown(f"**Card {idx + 1} of {total_fc}**")
 
                # Progress bar
                prog = round((idx / total_fc) * 100)
                st.markdown(
                    f'<div class="progress-bar-bg"><div class="progress-bar" style="width:{prog}%"></div></div>',
                    unsafe_allow_html=True
                )
 
                st.markdown("<br>", unsafe_allow_html=True)
 
                # Card display
                status = st.session_state.flashcard_status.get(idx, "")
                status_badge = ""
                if status == "got_it":
                    status_badge = '<span style="color:#059669;font-weight:700">✅ Got it!</span>'
                elif status == "review":
                    status_badge = '<span style="color:#dc2626;font-weight:700">🔄 Review again</span>'
 
                st.markdown(
                    f'<div class="flashcard-front">'
                    f'<div class="flashcard-front-text">{card.get("front", "")}</div>'
                    f'</div>'
                    f'<div class="flashcard-back">'
                    f'<div class="flashcard-back-text">{card.get("back", "")}</div>'
                    f'<div class="flashcard-topic">📌 {card.get("topic", "General")} &nbsp; {status_badge}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
 
                st.markdown("<br>", unsafe_allow_html=True)
 
                # Action buttons
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("← Previous", use_container_width=True):
                        st.session_state.current_fc_index = max(0, idx - 1)
                        st.rerun()
                with b2:
                    if st.button("✅ Got It", use_container_width=True):
                        st.session_state.flashcard_status[idx] = "got_it"
                        st.session_state.current_fc_index = min(total_fc - 1, idx + 1)
                        st.rerun()
                with b3:
                    if st.button("🔄 Review Again", use_container_width=True):
                        st.session_state.flashcard_status[idx] = "review"
                        st.session_state.current_fc_index = min(total_fc - 1, idx + 1)
                        st.rerun()
                with b4:
                    if st.button("Next →", use_container_width=True):
                        st.session_state.current_fc_index = min(total_fc - 1, idx + 1)
                        st.rerun()
 
                # Show cards needing review
                review_cards = [i for i, s in st.session_state.flashcard_status.items() if s == "review"]
                if review_cards:
                    st.markdown("---")
                    st.markdown(f"### 🔄 Cards to Review ({len(review_cards)})")
                    for i in review_cards:
                        c = flashcards[i]
                        st.markdown(
                            f'<div class="quiz-card" style="padding:12px 16px">'
                            f'<div style="color:#dc2626;font-size:13px;font-weight:600">{c.get("front", "")}</div>'
                            f'<div style="color:#64748b;font-size:12px;margin-top:6px">{c.get("back", "")}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
 
                if st.button("🔁 Reset All Cards", use_container_width=True):
                    st.session_state.flashcard_status  = {}
                    st.session_state.current_fc_index  = 0
                    st.rerun()
 
    else:
        st.markdown("### 👆 Upload a PDF to get started")
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            ("💬", "Study Chat", "Ask questions with voice input and get structured answers."),
            ("🧪", "Test Mode",  "Adaptive MCQ quizzes based on your performance."),
            ("📅", "Study Plan", "Personalized plan built from your PDF."),
            ("📈", "Analytics",  "Track topics, scores, streaks, and export reports."),
            ("🃏", "Flashcards", "Auto-generated spaced repetition cards from your answers."),
        ]
        for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5], cards):
            with col:
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div class="feature-icon">{icon}</div>'
                    f'<div class="feature-title">{title}</div>'
                    f'<div class="feature-desc">{desc}</div>'
                    f'</div>', unsafe_allow_html=True)
 
 
# ── ROUTER ────────────────────────────────────────
if st.session_state.user is None:
    show_auth_page()
else:
    show_main_app()