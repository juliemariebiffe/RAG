import os
import tempfile
import sqlite3

import streamlit as st
import pandas as pd

from rag.my_langchain import answer_question
from rag.my_langchain import delete_file_from_store
from rag.my_langchain import store_pdf_file

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# --- Base SQLite pour feedback et questions ---

def init_db():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note INTEGER NOT NULL,
            question TEXT NOT NULL,
            reponse TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            reponse TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_feedback(note: int, question: str, reponse: str):
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (note, question, reponse) VALUES (?, ?, ?)",
        (note, question, reponse)
    )
    conn.commit()
    conn.close()

def insert_question(question: str, reponse: str):
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO questions (question, reponse) VALUES (?, ?)",
        (question, reponse)
    )
    conn.commit()
    conn.close()

def get_all_feedbacks():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("SELECT id, note, question, reponse, date FROM feedback ORDER BY date DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_questions():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("SELECT id, question, reponse, date FROM questions ORDER BY date DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# --- Pages Streamlit ---

def analyse_page():
    if 'stored_files' not in st.session_state:
        st.session_state['stored_files'] = []

    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")
    
    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
        type=None,
        accept_multiple_files=True
    )
    
    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_in_kb:.2f}"
            })

            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, "temp.pdf")
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        delete_file_from_store(name)

    k = st.slider(
        label="Nombre de documents similaires à récupérer (k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    language = st.selectbox(
        "Choisissez la langue de réponse",
        options=["français", "anglais", "espagnol", "allemand"], 
        index=0
    )

    question = st.text_input("Votre question ici")

    if st.button("Analyser") and question.strip() != "":
        model_response = answer_question(question, language, k)
        st.text_area("Zone de texte, réponse du modèle", value=model_response, height=200)

        # Stocker question + réponse dans historique
        insert_question(question, model_response)

        feedback = st.radio(
            "Que pensez-vous de la qualité de la réponse ?",
            options=[1, 2, 3, 4, 5],
            index=4,
            format_func=lambda x: f"{x} étoiles",
            key="user_feedback"
        )

        if feedback is not None and model_response.strip() != "":
            insert_feedback(feedback, question, model_response)
            st.success("Merci pour votre feedback !")

def feedback_page():
    st.title("Consultation des feedbacks utilisateurs")

    feedbacks = get_all_feedbacks()

    if not feedbacks:
        st.info("Aucun feedback enregistré pour le moment.")
        return

    df = pd.DataFrame(feedbacks, columns=["ID", "Note", "Question", "Réponse", "Date"])

    notes_filter = st.multiselect("Filtrer par notes", options=[1,2,3,4,5], default=[1,2,3,4,5])
    if notes_filter:
        df = df[df["Note"].isin(notes_filter)]

    st.dataframe(df)

def questions_page():
    st.title("Historique des questions posées")

    questions = get_all_questions()

    if not questions:
        st.info("Aucune question enregistrée pour le moment.")
        return

    df = pd.DataFrame(questions, columns=["ID", "Question", "Réponse", "Date"])

    st.dataframe(df)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Analyse de documents", "Feedback utilisateurs", "Historique des questions"])

    if page == "Analyse de documents":
        analyse_page()
    elif page == "Feedback utilisateurs":
        feedback_page()
    elif page == "Historique des questions":
        questions_page()

if __name__ == "__main__":
    init_db()
    main()
