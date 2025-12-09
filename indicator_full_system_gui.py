import streamlit as st
from datetime import datetime
import sqlite3
import joblib
import pandas as pd

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "suggestion" not in st.session_state:
    st.session_state.suggestion = None
if "checked_word" not in st.session_state:
    st.session_state.checked_word = None
if "selected_level" not in st.session_state:
    st.session_state.selected_level = "remember"
if "level_page" not in st.session_state:
    st.session_state.level_page = None  # Track selected Bloom level

# ============================================
# LOAD MODELS
# ============================================
model = joblib.load("bloom_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ============================================
# LOAD BLOOM VERBS DATA
# ============================================
bloom_df = pd.read_csv("bloom_verbs.csv")
bloom_df['verb'] = bloom_df['verb'].astype(str).str.strip()
bloom_df['bloom_level'] = bloom_df['bloom_level'].astype(str).str.strip()
bloom_df = bloom_df[bloom_df['verb'] != ""]
bloom_df = bloom_df[bloom_df['bloom_level'] != ""]

def get_similar_verbs(level):
    level = level.strip().lower()
    verbs = bloom_df[bloom_df['bloom_level'].str.lower() == level]['verb'].tolist()
    return verbs

def display_verbs_table(verbs, cols=3):
    rows = [verbs[i:i+cols] for i in range(0, len(verbs), cols)]
    df = pd.DataFrame(rows)
    df = df.fillna("")
    df.columns = ["verbs"] + [" " * (i+1) for i in range(df.shape[1] - 1)]
    st.dataframe(df, hide_index=True, use_container_width=True)

# ============================================
# DATABASE SETUP
# ============================================
conn = sqlite3.connect("bloom_indicator.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bloom_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT,
    suggested_level TEXT,
    vote_count INTEGER DEFAULT 0,
    approved INTEGER DEFAULT 0,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    word_id INTEGER,
    UNIQUE(user_id, word_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT,
    predicted_level TEXT,
    created_at TEXT
)
""")

conn.commit()

# ============================================
# PREDICT QUESTION LEVEL
# ============================================
def predict_question(question):
    X = vectorizer.transform([question])
    pred = model.predict(X)[0]
    return pred

# ============================================
# CHECK / SUGGEST WORD
# ============================================
def check_or_predict_word(word):
    cursor.execute("""
        SELECT suggested_level FROM bloom_words 
        WHERE word=? AND approved=1
    """, (word,))
    result = cursor.fetchone()

    if result:
        return f"✅ '{word}' is already APPROVED at Bloom Level: {result[0]}"

    X = vectorizer.transform([word])
    pred = model.predict(X)[0]
    return f"ℹ NLP Suggestion: {pred}"

# ============================================
# SUBMIT / VOTE SYSTEM
# ============================================
def submit_word(word, level, user_id):

    if not user_id:
        return "Please enter your username before voting."

    cursor.execute("""
        SELECT id, vote_count FROM bloom_words 
        WHERE word=? AND suggested_level=?
    """, (word, level))
    result = cursor.fetchone()

    if result:
        word_id, vote_count = result

        cursor.execute("""
            SELECT id FROM votes 
            WHERE user_id=? AND word_id=?
        """, (user_id, word_id))

        if cursor.fetchone():
            return "You have already voted for this word."

        cursor.execute("INSERT INTO votes (user_id, word_id) VALUES (?, ?)", (user_id, word_id))

        vote_count += 1
        approved = 1 if vote_count >= 10 else 0

        cursor.execute("""
            UPDATE bloom_words
            SET vote_count=?, approved=?
            WHERE id=?
        """, (vote_count, approved, word_id))

        conn.commit()

        if approved:
            return f"🎉 '{word}' has reached 10 votes and is now APPROVED at level '{level}'."

        return f"✅ Vote recorded. '{word}' now has {vote_count} votes."

    cursor.execute("""
        INSERT INTO bloom_words (word, suggested_level, created_at, vote_count)
        VALUES (?, ?, ?, 1)
    """, (word, level, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    cursor.execute("SELECT last_insert_rowid()")
    word_id = cursor.fetchone()[0]

    cursor.execute("INSERT INTO votes (user_id, word_id) VALUES (?, ?)", (user_id, word_id))
    conn.commit()
    return f"✅ Added '{word}' and recorded your vote."

# ============================================
# STREAMLIT UI
# ============================================
st.title("🌿 Bloom’s Hybrid Indicator")

# USER LOGIN SECTION
st.sidebar.header("User Login")
username = st.sidebar.text_input("Enter your username")

if username:
    cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
    conn.commit()
    cursor.execute("SELECT id FROM users WHERE username=?", (username,))
    user_id = cursor.fetchone()[0]
else:
    user_id = None

menu = ["Predict Question", "Check / Submit Word", "Bloom’s Taxonomy Level"]
choice = st.sidebar.selectbox("Select Mode", menu)

# --------------------------------------------------
# MODE 1: Predict Question
# --------------------------------------------------
if choice == "Predict Question":
    st.header("🔍 Predict Bloom Level for a Question")
    question = st.text_area("Enter the full question:")

    if st.button("Predict"):
        if question.strip():
            pred = predict_question(question)
            st.success(f"Predicted Bloom Level: **{pred}**")

            similar_verbs = get_similar_verbs(pred)
            if similar_verbs:
                st.subheader(f"✨ Similar verbs for **{pred}**")
                display_verbs_table(similar_verbs, cols=3)

            cursor.execute("""
                INSERT INTO questions (question_text, predicted_level, created_at)
                VALUES (?, ?, ?)
            """, (question, pred, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
        else:
            st.warning("Please enter a question.")

# --------------------------------------------------
# MODE 2: Check / Submit Word
# --------------------------------------------------
elif choice == "Check / Submit Word":
    st.header("📝 Check or Submit a Word")
    word = st.text_input("Enter a verb or keyword:")

    if st.button("Check / Suggest"):
        if word.strip():
            st.session_state.suggestion = check_or_predict_word(word)
            st.session_state.checked_word = word
        else:
            st.warning("Enter a word first.")

    if st.session_state.suggestion:
        st.write(st.session_state.suggestion)

        if "Bloom Level:" in st.session_state.suggestion:
            suggested_level = st.session_state.suggestion.split("Bloom Level:")[1].strip()
        elif "NLP Suggestion:" in st.session_state.suggestion:
            suggested_level = st.session_state.suggestion.split("NLP Suggestion:")[1].strip()
        else:
            suggested_level = None

        if suggested_level:
            verbs = get_similar_verbs(suggested_level)
            if verbs:
                st.subheader(f"Similar verbs for {suggested_level}")
                display_verbs_table(verbs, cols=3)

        st.markdown("---")
        st.subheader("📌 Submit / Vote for Word")

        level = st.selectbox(
            "Suggested Bloom’s Level:",
            ["remember", "understand", "apply", "analyze", "evaluate", "create"],
            key="selected_level"
        )

        if st.button("Submit Word"):
            result = submit_word(st.session_state.checked_word, level, user_id)
            st.success(result)
            st.session_state.suggestion = None
            st.session_state.checked_word = None

# --------------------------------------------------
# MODE 3: Browse Levels (Box Layout)
# --------------------------------------------------
elif choice == "Bloom’s Taxonomy Level":
    st.header("📚 Browse Bloom Levels")

    # If a level is already selected, show verbs and return button
    if st.session_state.level_page:

        level = st.session_state.level_page
        st.subheader(f"✨ Verbs under {level.capitalize()}")
        verbs = get_similar_verbs(level)
        display_verbs_table(verbs, cols=3)

        if st.button("🔙 Return to Levels"):
            st.session_state.level_page = None  # Reset to show main level buttons

    # Else, show main Bloom level buttons in boxes
    else:
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        if col1.button("Remember", key="remember"):
            st.session_state.level_page = "remember"
        if col2.button("Understand", key="understand"):
            st.session_state.level_page = "understand"
        if col3.button("Apply", key="apply"):
            st.session_state.level_page = "apply"
        if col4.button("Analyze", key="analyze"):
            st.session_state.level_page = "analyze"
        if col5.button("Evaluate", key="evaluate"):
            st.session_state.level_page = "evaluate"
        if col6.button("Create", key="create"):
            st.session_state.level_page = "create"

