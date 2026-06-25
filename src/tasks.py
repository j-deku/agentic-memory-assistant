from database import get_connection
from insights import productivity_summary
from datetime import datetime
from collections import defaultdict
from collections import Counter
import pandas as pd
from learning_logger import log_event
from ml_model import predict_task
from predict_model import predict_task_completion
from task_to_ml_input import task_to_ml_input


def create_tasks_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT,
            due_date TEXT,

            completed INTEGER DEFAULT 0,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,

            estimated_minutes INTEGER,
            actual_minutes INTEGER,

            reminder_count INTEGER DEFAULT 0,

            completion_hour INTEGER,
            completion_day TEXT
        )
        """)

    conn.commit()
    conn.close()

def normalize_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except:
            print("[WARNING] Invalid date fixed automatically")
            return datetime.now().strftime("%Y-%m-%d")

def add_task(title, category, due_date):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO tasks (
            title,
            category,
            due_date
        )
        VALUES (?, ?, ?)
        """,
        (title, category, due_date)
    )

    conn.commit()
    conn.close()


def complete_task(task_id):
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()

    completion_hour = now.hour
    completion_day = now.strftime("%A")

    cursor.execute("""
        UPDATE tasks
        SET completed = 1,
            completed_at = ?,
            completion_hour = ?,
            completion_day = ?
        WHERE id = ?
    """, (now, completion_hour, completion_day, task_id))

    conn.commit()
    conn.close()

def create_prediction_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        predicted_prob REAL,
        actual_result INTEGER,
        error REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def update_prediction(task_id, actual_result):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE predictions
    SET actual_result = ?,
        error = ABS(predicted_prob - ?)
    WHERE task_id = ?
    """, (actual_result, actual_result, task_id))

    conn.commit()
    conn.close()

def log_prediction(task_id, predicted_prob):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (task_id, predicted_prob)
    VALUES (?, ?)
    """, (task_id, predicted_prob))

    conn.commit()
    conn.close()

# BEFORE — returns ALL tasks including completed
def list_tasks():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()
    return tasks

# AFTER — returns only pending tasks (completed = 0)
def list_tasks():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE completed = 0")
    tasks = cursor.fetchall()
    conn.close()
    return tasks

# ADD this alongside it for when you need everything
def list_all_tasks():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()
    return tasks

def delete_task(task_id):
    conn = get_connection()
    cursor = conn.cursor()

    # Check if task exists
    cursor.execute(
        "SELECT title FROM tasks WHERE id = ?",
        (task_id,)
    )

    task = cursor.fetchone()

    if not task:
        conn.close()
        return f"Task {task_id} not found."

    task_title = task[0]

    # Delete task
    cursor.execute(
        "DELETE FROM tasks WHERE id = ?",
        (task_id,)
    )

    conn.commit()
    conn.close()

    return f"🗑️ Deleted task: {task_title}"

def get_overdue_tasks():
    conn = get_connection()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT *
        FROM tasks
        WHERE completed = 0
        AND due_date < ?
    """, (today,))

    tasks = cursor.fetchall()

    conn.close()

    return tasks


def get_recommended_tasks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM tasks
        WHERE completed = 0
        ORDER BY due_date ASC
    """)

    tasks = cursor.fetchall()

    conn.close()

    return tasks


def get_task_priority(task):
    due_date = datetime.strptime(task[3], "%Y-%m-%d").date()
    today = datetime.now().date()

    days_left = (due_date - today).days

    if days_left < 0:
        return "🔴 OVERDUE"

    elif days_left <= 3:
        return "🟡 DUE SOON"

    else:
        return "🟢 UPCOMING"



def get_task_score(task):
    today = datetime.now().date()
 
    category = (task[2] or "other").lower()
 
    # Guard: tasks with no due_date get a neutral urgency score
    if not task[3]:
        days_left = 99
    else:
        try:
            due_date = datetime.strptime(task[3], "%Y-%m-%d").date()
            days_left = (due_date - today).days
        except ValueError:
            days_left = 99
 
    score = 0
    reasons = []
 
    if days_left < 0:
        score += 100
        reasons.append("Overdue")
    elif days_left <= 2:
        score += 70
        reasons.append("Due very soon")
    elif days_left <= 5:
        score += 40
        reasons.append("Due soon")
    else:
        score += 10
        reasons.append("Not urgent")
 
    category_scores = {
        "work": (25, "Work task"),
        "school": (20, "School task"),
        "study": (20, "Study task"),
        "health": (15, "Health task"),
        "learning": (10, "Learning task"),
    }
    cat_score, cat_reason = category_scores.get(category, (5, "Personal task"))
    score += cat_score
    reasons.append(cat_reason)
 
    return score, reasons


def get_daily_briefing():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()

    conn.close()

    if not tasks:
        return None

    return tasks


def generate_briefing(tasks, get_task_score):
        if not tasks:
            return "No tasks for today. 🎉"

        # sort by priority score
        ranked = sorted(tasks, key=lambda t: get_task_score(t)[0], reverse=True)

        top_task = ranked[0]
        top_score, top_reason = get_task_score(top_task)

        overdue = []
        due_soon = []

        today = datetime.now().date()

        for t in tasks:
            due_date = datetime.strptime(t[3], "%Y-%m-%d").date()
            days_left = (due_date - today).days

            if t[4] == 0:
                if days_left < 0:
                    overdue.append(t)
                elif days_left <= 2:
                    due_soon.append(t)

        summary = productivity_summary()

        report = f"""
    📊 DAILY BRIEFING

    🔥 Top Priority Task:
    {top_task[1]} (Score: {top_score})
    Reason: {", ".join(top_reason)}

    📌 Summary:
    {summary}

    ⚠ Overdue Tasks: {len(overdue)}
    📅 Due Soon: {len(due_soon)}
    """

        return report


def analyze_habits():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()

    conn.close()

    if not tasks:
        return "No data for habit analysis yet."

    category_stats = defaultdict(lambda: {"total": 0, "completed": 0})

    for t in tasks:
        category = t[2]

        category_stats[category]["total"] += 1

        if t[4] == 1:
            category_stats[category]["completed"] += 1

    report = "\n🧠 HABIT INSIGHTS\n"

    for category, data in category_stats.items():
        total = data["total"]
        completed = data["completed"]
        rate = (completed / total) * 100

        report += f"""
    📌 {category}
    - Total tasks: {total}
    - Completed: {completed}
    - Completion rate: {rate:.1f}%
    """

    return report


def productivity_score():
    from database import get_connection
    conn = get_connection()
    cursor = conn.cursor() 
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()
 
    if not tasks:
        return "No data available."
 
    total = len(tasks)
    completed = sum(t[4] for t in tasks)
    completion_rate = completed / total * 100
 
    today = datetime.now().date()
    overdue = 0
    recent_activity = 0
 
    for t in tasks:
        # Guard: skip tasks with no due_date
        if not t[3]:
            continue
        try:
            due_date = datetime.strptime(t[3], "%Y-%m-%d").date()
        except ValueError:
            continue
 
        if t[4] == 0 and due_date < today:
            overdue += 1
        if (today - due_date).days <= 3:
            recent_activity += 1
 
    score = 0

    # base productivity
    score += completion_rate * 0.7

    # overdue penalty (soft cap)
    score -= min(overdue * 3, 25)

    # recent activity reward
    score += min(recent_activity * 2, 20)

    # clamp
    score = max(0, min(100, score))
 
    if score >= 80:
        status = "Excellent 🚀"
    elif score >= 60:
        status = "Good 👍"
    elif score >= 40:
        status = "Average ⚠"
    else:
        status = "Needs Improvement ❌"
 
    return (
        f"📊 Productivity Score: {score:.1f} / 100\n"
        f"  Completion rate: {completion_rate:.1f}%\n"
        f"  Overdue tasks: {overdue}\n"
        f"  Recent activity boost: +{min(recent_activity * 2, 20)}\n"
        f"  Status: {status}"
    )

def create_memory_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE,
        value TEXT
    )
    """)

    conn.commit()
    conn.close()

def create_prediction_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        predicted_prob REAL,
        actual_result INTEGER,
        error REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def create_learning_events_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS learning_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        event_type TEXT,
        reward REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def set_memory(key, value):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO memory (key, value)
    VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))

    conn.commit()
    conn.close()


def get_memory(key):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM memory WHERE key = ?", (key,))
    result = cursor.fetchone()

    conn.close()

    return result[0] if result else None


def update_memory_from_tasks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()

    conn.close()

    if not tasks:
        return

    categories = [t[2] for t in tasks]
    completed = sum(t[4] for t in tasks)

    most_common_category = Counter(categories).most_common(1)[0][0]
    total_tasks = len(tasks)
    completion_rate = (completed / total_tasks) * 100

    set_memory("most_used_category", most_common_category)
    set_memory("completion_rate", str(completion_rate))
    set_memory("total_tasks", str(total_tasks))


def predictive_insights():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()

    conn.close()

    if not tasks:
        return "No data for predictions yet."

    total = len(tasks)
    completed = sum(t[4] for t in tasks)

    today = datetime.now().date()

    overdue = 0
    personal_delays = 0
    work_completion = 0
    work_total = 0

    for t in tasks:
        category = t[2].lower()
        due_date = datetime.strptime(t[3], "%Y-%m-%d").date()

        if t[4] == 0 and due_date < today:
            overdue += 1

        # category behavior tracking
        if category == "personal":
            if t[4] == 0 and due_date < today:
                personal_delays += 1

        if category == "work":
            work_total += 1
            if t[4] == 1:
                work_completion += 1

    completion_rate = (completed / total) * 100

    work_rate = (work_completion / work_total * 100) if work_total > 0 else 0

    prediction = []

    # --- Behavior rules (simple predictive logic) ---

    if personal_delays >= 2:
        prediction.append("You tend to delay personal tasks ⚠")

    if completion_rate < 50:
        prediction.append("Low completion trend detected 📉")

    if work_rate > 70:
        prediction.append("Strong work performance pattern 💼")

    if overdue >= 2:
        prediction.append("High risk of task accumulation 🔴")

    if completion_rate > 70:
        prediction.append("Productivity improving 📈")

    if not prediction:
        prediction.append("Stable behavior pattern detected ✅")

    return "\n".join(prediction)


def build_ml_dataset():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()

    conn.close()

    if not tasks:
        return None

    data = []

    today = datetime.now().date()

    for t in tasks:
        task_id = t[0]
        title = t[1]
        category = t[2]
        due_date = datetime.strptime(t[3], "%Y-%m-%d").date()
        completed = t[4]

        days_left = (due_date - today).days
        is_overdue = 1 if days_left < 0 else 0

        # FEATURE ENGINEERING (VERY IMPORTANT)
        row = {
            "task_length": len(title),
            "category": category.lower(),
            "days_left": days_left,
            "is_overdue": is_overdue,
            "label_completed": completed
        }

        data.append(row)

    df = pd.DataFrame(data)
    return df


def export_ml_dataset(path="ml_tasks_dataset.csv"):
    df = build_ml_dataset()

    if df is None:
        print("No data to export.")
        return

    df.to_csv(path, index=False)
    print(f"Dataset exported to {path}")

def ml_predict_for_task(task):
    from datetime import datetime

    title = task[1]
    category = task[2]
    due_date = datetime.strptime(task[3], "%Y-%m-%d").date()

    days_left = (due_date - datetime.now().date()).days
    is_overdue = 1 if days_left < 0 else 0

    prediction, probability = predict_task(
        task_length=len(title),
        category=category.lower(),
        days_left=days_left,
        is_overdue=is_overdue
    )

    return prediction, probability

def safe_predict(task):
    try:
        return predict_task_completion(
            task_to_ml_input(task)
        )
    except Exception as e:
        return None

def log_negative_feedback(task_id):
    update_prediction(task_id, 0)
    log_event(task_id, "task_failed_or_delayed", -10)



def show_schema():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(tasks)")

    for row in cursor.fetchall():
        print(row)

    conn.close()
