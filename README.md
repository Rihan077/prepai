🎓 PrepAI — AI-Powered Adaptive Study Assistant


Turn any PDF into a complete intelligent learning system with RAG, hybrid search, adaptive quizzes, concept maps, timed exams, and persistent user memory.



Show Image
Show Image
Show Image
Show Image
Show Image


🚀 Live Demo

Try PrepAI Live →


📌 What Problem Does It Solve?

Students have PDFs, notes, and textbooks but no intelligent way to study from them. PrepAI transforms any PDF into a complete adaptive learning platform — it answers questions, generates quizzes, builds personalized study plans, creates flashcards, and tracks your learning progress across sessions.


✨ Features

💬 Study Chat


Ask questions from your uploaded PDF(s) and get structured 2 Mark / 5 Mark / 10 Mark answers
Hybrid retrieval combining BM25 keyword search + FAISS semantic search using Reciprocal Rank Fusion
Every answer evaluated on 3 quality metrics: Faithfulness, Answer Relevancy, Context Precision
Conversation memory — system remembers previous questions in the session
Voice input via OpenAI Whisper — speak your question instead of typing
Source attribution showing exactly which PDF and which page the answer came from


🧪 Test Mode (Adaptive Quiz)


MCQs generated as structured JSON — no fragile string parsing
Adaptive difficulty — system automatically adjusts to Easy/Medium/Hard based on your past quiz scores
Hints shown on Easy questions, tricky distractors on Hard questions
Full answer review with explanations after submission


📅 Study Plan


AI analyzes your PDF and distributes topics across N days intelligently
Weak topic prioritization — topics you struggle with get scheduled first
Day cards with topics, focus statement, suggested questions, difficulty, estimated hours
Mark days complete with persistent progress bar (saved to database)


📈 Analytics Dashboard


Weak topic bar chart with severity levels (red/yellow/navy)
Quiz score history line chart with adaptive difficulty color coding
Recent questions log with confidence scores
Day streak tracking — study consecutive days to build streaks
Export full session report as downloadable file


🃏 Flashcards


Auto-generated after every question using spaced repetition principles
Front (concept) + Back (explanation) + Topic tag
Got It / Review Again tracking
Cards needing review surfaced at bottom for re-study


🗺️ Concept Map


Visual network graph showing how topics in your PDF connect
Nodes color-coded: Navy = Core Topic, Gold = Subtopic, Grey = Detail
Relationship labels: includes, requires, leads to, part of
Works across multiple PDFs simultaneously


📝 Exam Mode


Full timed mock exam — choose MCQ count, short answer count, time limit, difficulty
Countdown timer turns red under 2 minutes — auto-submits when time runs out
Section A: MCQs with adaptive difficulty
Section B: Short answer questions with model answers and key points
Complete results with grade (Distinction / Pass / Fail)


📚 Multi-PDF Support


Upload up to 5 PDFs simultaneously
Cross-document retrieval — questions search across ALL PDFs at once
Per-PDF Document Intelligence — summary, topics, difficulty, study time
Source attribution shows exactly which PDF and pages contributed to each answer
Remove individual PDFs without losing others


🔐 Authentication + Persistent Memory


Login / Register with bcrypt password hashing
SQLite database — all data persists across sessions and logouts
Level 1: Session persistence (survives page refresh)
Level 2: User profiles with full history
Level 3: Adaptive behavior based on historical performance



🏗️ Architecture

PDF Upload(s)
      ↓
Text Extraction (PyPDFLoader)
      ↓
Chunking (RecursiveCharacterTextSplitter)
      ↓
┌─────────────────────────────┐
│  FAISS (Semantic Search)    │
│  BM25  (Keyword Search)     │
└──────────┬──────────────────┘
           ↓
  Reciprocal Rank Fusion (RRF)
           ↓
  GPT-4o-mini (Answer Generation)
           ↓
  RAGAS / Local Evaluation
           ↓
  Response + Metrics + Memory


🛠️ Tech Stack

TechnologyPurposePython 3.11Core languageStreamlitWeb UI frameworkLangChainRAG pipeline orchestrationOpenAI GPT-4o-miniAnswer generation, quiz/plan/exam creation, topic detectionOpenAI WhisperVoice input transcriptionOpenAI EmbeddingsText → vector conversionFAISSVector database for semantic searchBM25 (rank-bm25)Keyword search indexReciprocal Rank FusionHybrid retrieval result mergingRAGASProduction-grade RAG evaluationSQLitePersistent user data storagebcryptPassword hashingPlotlyInteractive analytics charts + concept mappython-dotenvSecure API key management


📁 Project Structure

prepai/
├── app.py              # Main Streamlit application (7 modes)
├── rag_pipeline.py     # RAG engine, hybrid search, all AI functions
├── auth.py             # Login, register, password hashing
├── database.py         # SQLite setup, all read/write functions
├── requirements.txt    # Python dependencies
└── .env                # API keys (not committed)


⚙️ Setup & Installation

1. Clone the repository

bashgit clone https://github.com/Rihan077/prepai.git
cd prepai

2. Create virtual environment

bashpython -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

3. Install dependencies

bashpip install -r requirements.txt

4. Set up environment variables

Create a .env file in the root directory:

OPENAI_API_KEY=your-openai-api-key-here

5. Run the app

bashstreamlit run app.py


🧠 Key Technical Concepts

Hybrid Search (BM25 + FAISS + RRF)

Most RAG projects use only semantic search. PrepAI combines keyword search (BM25 — same algorithm as Elasticsearch) and semantic search (FAISS) using Reciprocal Rank Fusion. Formula: score(doc) = Σ 1/(k + rank) where k=60 is the standard constant from the original RRF research paper.

RAGAS Evaluation Pipeline

Every answer is automatically scored on three industry-standard metrics:


Faithfulness — is the answer grounded in the source document?
Answer Relevancy — does it actually answer what was asked?
Context Precision — did retrieval fetch the right chunks?


Adaptive Difficulty Engine

Quiz difficulty adjusts based on performance history stored in SQLite. Average score < 50% on a topic → Easy questions with hints. Average score > 80% → Hard questions with tricky distractors. This is how real EdTech platforms like Duolingo work.

Multi-PDF Cross-Document Retrieval

Each PDF gets its own FAISS + BM25 index. On query, all indexes are searched in parallel. Results are merged using RRF across all sources. Each retrieved chunk is tagged with its source PDF and page number, enabling precise attribution.


🎯 Interview One-Liner


"I built PrepAI — an adaptive learning platform with hybrid BM25+FAISS retrieval, RAGAS evaluation, adaptive quiz difficulty, multi-PDF cross-document search, voice input, concept maps, timed exams, and persistent user memory using SQLite — deployed on Streamlit Cloud."




📊 What Makes This Different

FeatureMost Student RAG ProjectsPrepAISearchSemantic onlyHybrid BM25 + FAISS + RRFEvaluationNoneRAGAS (Faithfulness, Relevancy, Precision)QuizBasic MCQAdaptive difficulty based on historyMemorySession onlyPersistent SQLite across sessionsMulti-documentSingle PDFUp to 5 PDFs simultaneouslyVoiceNoneOpenAI Whisper integrationConcept MapNonePlotly network graphExamNoneTimed with auto-submit + model answersAuthNoneLogin + bcrypt password hashing


👤 Author

Rihan Bagwan
B.Tech CSE (AI & Data Science) — Sanjay Ghodawat University
IIT Roorkee IntelliPath AI & DS Programme

Show Image
Show Image


📄 License

MIT License — feel free to use, modify and distribute.
