# 🎓 PrepAI – AI-Powered Adaptive Study Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

An intelligent study assistant that allows students to upload PDFs and learn smarter using AI-powered question answering, test generation, personalized study plans, and learning analytics.

</div>

---

# 📚 Overview

PrepAI is an AI-powered adaptive learning platform built using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

Students can upload their study materials and instantly:

✅ Ask questions from their notes

✅ Generate quizzes and tests

✅ Create personalized study plans

✅ Track weak topics and learning progress

The goal of PrepAI is to make learning **personalized, interactive, and efficient**.

---

# ✨ Features

## 💬 Study Chat
- Ask questions directly from uploaded PDFs.
- Get structured answers.
- Supports short and long-form responses.

## 📝 Test Mode
- Generate MCQs automatically.
- Instant scoring and explanations.
- Helps with exam preparation.

## 📅 Study Plan
- Generates day-by-day study schedules.
- Personalized according to uploaded material.

## 📊 Analytics
- Track weak topics.
- Monitor progress and quiz performance.
- Identify areas that need improvement.

## 🔍 Hybrid Search
- Semantic search using vector embeddings.
- Context-aware retrieval.
- Accurate responses from study material.

---

# 🏗️ System Architecture

```text
User Uploads PDF
        ↓
PDF Processing
        ↓
Chunking & Embeddings
        ↓
Vector Database
        ↓
RAG Pipeline
        ↓
GPT-4o-mini
        ↓
Answer Generation
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend |
| Streamlit | Web Application |
| OpenAI GPT-4o-mini | LLM |
| RAG Pipeline | Context Retrieval |
| Vector Database | Semantic Search |
| PyPDF | PDF Processing |
| GitHub | Version Control |

---

# 📂 Project Structure

```bash
prepai/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── temp.pdf
└── README.md
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Rihan077/prepai.git
cd prepai
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Add Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

### Run Application

```bash
streamlit run app.py
```

---

# 🌐 Live Demo

Coming Soon...

---

# 📸 Application Preview

- Upload PDFs
- Ask questions
- Generate tests
- Create study plans
- Analyze weak topics

---

# 🎯 Future Roadmap

- User Authentication
- Multi-document Support
- Voice Assistant
- AI Tutor Mode
- Flashcard Generation
- Leaderboards
- Mobile Application
- Multi-language Support
- Cloud Deployment
- AI Agents Integration

---

# 💡 Use Cases

- College Students
- Competitive Exam Preparation
- Self-Learning
- Online Courses
- Teachers and Educators
- Corporate Training

---

# 🔒 Security

- API keys stored securely using environment variables.
- No permanent storage of uploaded documents.
- User data privacy focused.

---

# 📈 Future Vision

PrepAI aims to become a complete AI learning ecosystem that combines:

- Personalized Learning
- Intelligent Tutoring
- Adaptive Assessments
- Learning Analytics
- AI Agents for Education

---

# 👨‍💻 Developer

**Rihan Bagwan**

B.Tech Computer Science Engineering  
AI & Data Science Enthusiast  
Building AI-powered educational products and intelligent systems.

GitHub: https://github.com/Rihan077

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
📢 Share it with others

---

# 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### 🚀 Learn Smarter with PrepAI

AI-powered adaptive learning for every student.

</div>
