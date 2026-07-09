import sqlite3
import os
from datetime import datetime, date

DB_PATH = "prepai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS weak_topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        count INTEGER DEFAULT 1,
        last_seen TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS quiz_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT,
        score INTEGER,
        total INTEGER,
        percentage INTEGER,
        difficulty TEXT DEFAULT 'Medium',
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS question_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT,
        topic TEXT,
        confidence REAL,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_date TEXT,
        questions_asked INTEGER DEFAULT 0,
        quizzes_taken INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS study_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        plan_title TEXT,
        plan_data TEXT,
        completed_days TEXT DEFAULT '[]',
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")

    conn.commit()
    conn.close()


# ── USER FUNCTIONS ────────────────────────────────
def create_user(username, password_hash):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user


def update_last_login(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(), user_id)
    )
    conn.commit()
    conn.close()


# ── WEAK TOPICS ───────────────────────────────────
def save_topic(user_id, topic):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, count FROM weak_topics WHERE user_id = ? AND topic = ?",
        (user_id, topic)
    )
    existing = c.fetchone()
    if existing:
        c.execute(
            "UPDATE weak_topics SET count = ?, last_seen = ? WHERE id = ?",
            (existing[1] + 1, datetime.now().isoformat(), existing[0])
        )
    else:
        c.execute(
            "INSERT INTO weak_topics (user_id, topic, count, last_seen) VALUES (?, ?, 1, ?)",
            (user_id, topic, datetime.now().isoformat())
        )
    conn.commit()
    conn.close()


def get_topics(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT topic, count FROM weak_topics WHERE user_id = ? ORDER BY count DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


# ── QUIZ HISTORY ──────────────────────────────────
def save_quiz(user_id, topic, score, total, percentage, difficulty="Medium"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO quiz_history (user_id, topic, score, total, percentage, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, topic, score, total, percentage, difficulty, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_quiz_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT topic, score, total, percentage, difficulty, created_at FROM quiz_history WHERE user_id = ? ORDER BY created_at ASC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"topic": r[0], "score": r[1], "total": r[2],
         "pct": r[3], "difficulty": r[4], "time": r[5][:16]}
        for r in rows
    ]


def get_adaptive_difficulty(user_id, topic):
    """
    Level 3 — Adaptive difficulty based on past performance.
    Below 50% twice → Easy
    Above 80% twice → Hard
    Otherwise → Medium
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT percentage FROM quiz_history WHERE user_id = ? AND topic = ? ORDER BY created_at DESC LIMIT 3",
        (user_id, topic)
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return "Medium"

    scores = [r[0] for r in rows]
    avg = sum(scores) / len(scores)

    if avg < 50:
        return "Easy"
    elif avg > 80:
        return "Hard"
    return "Medium"


# ── QUESTION LOG ──────────────────────────────────
def save_question(user_id, query, topic, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO question_log (user_id, query, topic, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, query, topic, confidence, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_question_log(user_id, limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT query, topic, confidence, created_at FROM question_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"query": r[0], "topic": r[1],
         "confidence": r[2], "time": r[3][11:16]}
        for r in rows
    ]


# ── STREAK TRACKING ───────────────────────────────
def update_streak(user_id):
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id FROM study_sessions WHERE user_id = ? AND session_date = ?",
        (user_id, today)
    )
    existing = c.fetchone()
    if not existing:
        c.execute(
            "INSERT INTO study_sessions (user_id, session_date, questions_asked, quizzes_taken) VALUES (?, ?, 0, 0)",
            (user_id, today)
        )
    conn.commit()
    conn.close()


def get_streak(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT session_date FROM study_sessions WHERE user_id = ? ORDER BY session_date DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return 0

    streak = 0
    check_date = date.today()
    for row in rows:
        session_date = date.fromisoformat(row[0])
        if session_date == check_date:
            streak += 1
            check_date = date.fromordinal(check_date.toordinal() - 1)
        else:
            break
    return streak


def get_total_study_days(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM study_sessions WHERE user_id = ?",
        (user_id,)
    )
    count = c.fetchone()[0]
    conn.close()
    return count


# ── STUDY PLAN ────────────────────────────────────
def save_study_plan(user_id, plan_title, plan_data):
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO study_plans (user_id, plan_title, plan_data, completed_days, created_at) VALUES (?, ?, ?, '[]', ?)",
        (user_id, plan_title, json.dumps(plan_data), datetime.now().isoformat())
    )
    plan_id = c.lastrowid
    conn.commit()
    conn.close()
    return plan_id


def get_latest_study_plan(user_id):
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, plan_title, plan_data, completed_days FROM study_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user_id,)
    )
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "plan_title": row[1],
            "plan_data": json.loads(row[2]),
            "completed_days": set(json.loads(row[3]))
        }
    return None


def update_completed_days(plan_id, completed_days):
    import json
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE study_plans SET completed_days = ? WHERE id = ?",
        (json.dumps(list(completed_days)), plan_id)
    )
    conn.commit()
    conn.close()


# ── ADAPTIVE RETRIEVAL COUNT ──────────────────────
def get_adaptive_k(user_id):
    """
    Level 3 — If user consistently gets low confidence scores,
    retrieve more chunks automatically.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT AVG(confidence) FROM question_log WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
        (user_id,)
    )
    avg = c.fetchone()[0]
    conn.close()

    if avg is None:
        return 4
    if avg < 50:
        return 6
    if avg < 70:
        return 5
    return 4