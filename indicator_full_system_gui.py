import streamlit as st
from datetime import datetime
import sqlite3
import joblib
import pandas as pd

# ============================================
# INITIALIZE SESSION STATE (Fix UI glitches)
# ============================================
if "suggestion" not in st.session_state:
    st.session_state.suggestion = None
if "checked_word" not in st.session_state:
    st.session_state.checked_word = None
if "selected_level" not in st.session_state:
    st.session_state.selected_level = "remember"

# -------------------------------------------
# LOAD MODELS
# -------------------------------------------
model = joblib.load("bloom_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------------------------------------------
# LOAD BLOOM VERBS CSV
# -------------------------------------------
bloom_df = pd.read_csv("bloom_verbs.csv")
bloom_df['verb'] = bloom_df['verb'].astype(str).str.strip()
bloom_df['bloom_level'] = bloom_df['bloom_level'].astype(str).str.strip()
bloom_df = bloom_df[bloom_df['verb'] != ""]
bloom_df = bloom_df[bloom_df['bloom_level'] != ""]

# -------------------------------------------
# FUNCTION TO GET SIMILAR VERBS
# -------------------------------------------
def get_similar_verbs(level):
    level_clean = level.strip().lower()
    verbs = bloom_df[bloom_df['bloom_level'].str.lower() == level_clean]['verb'].tolist()
    return verbs

# -------------------------------------------
# FUNCTION TO DISPLAY VERBS IN TABLE (UPDATED)
# -------------------------------------------
def display_verbs_table(verbs, cols=3):
    rows = [verbs[i:i+cols] for i in range(0, len(verbs), cols)]
    df = pd.DataFrame(rows)
    df = df.fillna("")

    # Unique but visually empty column names
    df.columns = ["verbs"] + [" " * (i+1) for i in range(df.shape[1] - 1)]

    st.dataframe(df, hide_index=True, use_container_width=True)


# -------------------------------------------
# CONNECT TO DATABASE
# -------------------------------------------
conn = sqlite3.connect("bloom_indicator.db", check_same_thread=False)
cursor = conn.cursor()

# -------------------------------------------
# PREDICT BLOOM LEVEL FOR QUESTION
# -------------------------------------------
def predict_question(question):
    X = vectorizer.transform([question])
    pred = model.predict(X)[0]
    return pred

# -------------------------------------------
# SUBMIT WORD TO DATABASE
# -------------------------------------------
def submit_word(word, level):
    cursor.execute("""
        SELECT id, vote_count FROM bloom_words 
        WHERE word=? AND suggested_level=?
    """, (word, level))
    
    result = cursor.fetchone()

    if result:
        word_id, vote_count = result
        vote_count += 1
        approved = vote_count >= 10

        cursor.execute("""
            UPDATE bloom_words 
            SET vote_count=?, approved=? 
            WHERE id=?
        """, (vote_count, approved, word_id))
        conn.commit()

        return f"✅ Updated '{word}' | Level '{level}' | Votes: {vote_count} | Approved: {approved}"

    cursor.execute("""
        INSERT INTO bloom_words (word, suggested_level, created_at) 
        VALUES (?, ?, ?)
    """, (word, level, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

    return f"✅ Added new word '{word}' under suggested level '{level}'"

# -------------------------------------------
# CHECK OR SUGGEST WORD LEVEL
# -------------------------------------------
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

# -------------------------------------------
# STREAMLIT UI
# -------------------------------------------
st.title("🌿 Bloom’s Hybrid Indicator System")

menu = ["Predict Question", "Check / Submit Word", "Bloom’s Taxonomy Level"]
choice = st.sidebar.selectbox("Select Mode", menu)

# --------------------------------------------------
# MODE 1: Predict Full Question
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
            else:
                st.info("No verbs found for this level.")

            cursor.execute("""
                INSERT INTO questions (question_text, predicted_level, created_at) 
                VALUES (?, ?, ?)
            """, (question, pred, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

            st.write("💾 Question saved to database.")

        else:
            st.warning("Please enter a question.")

# --------------------------------------------------
# MODE 2: Check, Suggest, Submit Word
# --------------------------------------------------
elif choice == "Check / Submit Word":
    st.header("📝 Check or Submit a Word to Bloom Level")

    word = st.text_input("Enter a verb or keyword:", key="input_word")

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
            similar_verbs = get_similar_verbs(suggested_level)
            if similar_verbs:
                st.markdown(f"**Similar verbs for {suggested_level}:**")
                display_verbs_table(similar_verbs, cols=3)
            else:
                st.info("No verbs found for this level.")

        st.markdown("---")
        st.subheader("📌 Submit / Vote for This Word")

        level = st.selectbox(
            "Suggested Bloom’s Level:",
            ["remember", "understand", "apply", "analyze", "evaluate", "create"],
            key="selected_level"
        )

        if st.button("Submit Word"):
            result = submit_word(st.session_state.checked_word, level)
            st.success(result)

            st.session_state.suggestion = None
            st.session_state.checked_word = None

# --------------------------------------------------
# MODE 3: Browse Bloom’s Taxonomy Levels
# --------------------------------------------------
elif choice == "Bloom’s Taxonomy Level":
    st.header("📚 Browse Bloom’s Taxonomy Levels")
    st.write("Click a level to view all verbs under it.")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    if col1.button("Remember"):
        level = "remember"
    elif col2.button("Understand"):
        level = "understand"
    elif col3.button("Apply"):
        level = "apply"
    elif col4.button("Analyze"):
        level = "analyze"
    elif col5.button("Evaluate"):
        level = "evaluate"
    elif col6.button("Create"):
        level = "create"
    else:
        level = None

    if level:
        st.subheader(f"✨ Verbs under **{level.capitalize()}**")

        verbs = get_similar_verbs(level)
        if verbs:
            display_verbs_table(verbs, cols=3)
        else:
            st.info("No verbs found for this level.")

