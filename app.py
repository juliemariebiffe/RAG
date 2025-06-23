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

# Initialisation de la base SQLite pour le feedback
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
    conn.commit()
    conn.close()

# Fonction pour insérer un feedback dans la base
def insert_feedback(note: int, question: str, reponse: str):
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (note, question, reponse) VALUES (?, ?, ?)",
        (note, question, reponse)
    )
    conn.commit()
    conn.close()

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

def main():
    # Titre et explications
    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")
    
    # Téléversement de fichiers multiples
    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
        type=None,  # ou ['pdf', 'txt', 'docx', ...] selon vos besoins
        accept_multiple_files=True
    )
    
    # S'il y a des fichiers, on affiche leurs noms et tailles
    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            # La taille, en octets, se récupère via len(f.getvalue())
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

    # Gestion de la suppression de documents
    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        delete_file_from_store(name)


    # Sélecteur du nombre de documents similaires à récupérer
    k = st.slider(
        label="Nombre de documents similaires à récupérer (k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Sélecteur de langue
    language = st.selectbox(
        "Choisissez la langue de réponse",
        options=["français", "anglais", "espagnol", "allemand"], 
        index=0
    )

    # Champ de question
    question = st.text_input("Votre question ici")

    # Bouton pour lancer l’analyse
    if st.button("Analyser") and question.strip() != "":
        model_response = answer_question(question, language, k)  # on ajoute k ici
        st.text_area("Zone de texte, réponse du modèle", value=model_response, height=200)

        # Notation de la réponse avec st.radio
        feedback = st.radio(
            "Que pensez-vous de la qualité de la réponse ?",
            options=[1, 2, 3, 4, 5],
            index=4,
            format_func=lambda x: f"{x} étoiles",
            key="user_feedback"
        )

        # Enregistrement du feedback dans la base SQLite
        if feedback is not None and model_response.strip() != "":
            insert_feedback(feedback, question, model_response)
            st.success("Merci pour votre feedback !")

if __name__ == "__main__":
    init_db()
    main()
