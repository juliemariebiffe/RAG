import os
import sqlite3
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime

from rag.my_langchain import answer_question
from rag.my_langchain import delete_file_from_store
from rag.my_langchain import store_pdf_file

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# Connexion à la base SQLite
DB_PATH = "feedback.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Création des tables si elles n'existent pas
def init_db():
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            response TEXT,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

init_db()

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def main():
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

    if st.button("Analyser"):
        if question.strip() != "":
            model_response = answer_question(question, language, k)

            st.text_area("Réponse du modèle", value=model_response, height=200)

            st.session_state['chat_history'].append((question, model_response))

            # Enregistrement dans la table questions
            c.execute("INSERT INTO questions (question) VALUES (?)", (question,))
            conn.commit()

            # Feedback étoilé
            feedback = st.radio(
                "Que pensez-vous de la qualité de la réponse ?",
                options=[1, 2, 3, 4, 5],
                index=4,
                format_func=lambda x: f"{x} étoile{'s' if x > 1 else ''}"
            )
            comment = st.text_input("Un commentaire pour améliorer la réponse ? (optionnel)")

            if st.button("Envoyer le feedback"):
                c.execute("""
                    INSERT INTO feedbacks (question, response, rating, comment)
                    VALUES (?, ?, ?, ?)
                """, (question, model_response, feedback, comment))
                conn.commit()
                st.success("Merci pour votre retour !")

    # Historique de chat
    if st.session_state['chat_history']:
        st.markdown("## Historique de la session")
        for i, (q, r) in enumerate(reversed(st.session_state['chat_history'])):
            with st.expander(f"Question {len(st.session_state['chat_history']) - i}"):
                st.markdown(f"**Q :** {q}")
                st.markdown(f"**R :** {r}")

if __name__ == "__main__":
    main()
