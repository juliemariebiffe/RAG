import os
import tempfile

import streamlit as st
import pandas as pd

from rag import my_langchain
from rag import my_llamaindex  # On importe le module, pas une classe

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# Initialisation des états
if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []
if 'framework' not in st.session_state:
    st.session_state['framework'] = "langchain"


def clear_indexes():
    my_langchain.clear_index()
    # Pas de fonction clear_index dans my_llamaindex.py pour l’instant
    # Si tu veux, tu peux ajouter une fonction clear_index dans my_llamaindex.py
    # Ici on laisse vide ou on peut réinitialiser le vector_store si tu modifies my_llamaindex.py


def main():
    st.title("Analyse de documents")
    st.subheader("Analysez vos documents avec une IA en les chargeant dans l'application. Puis posez toutes vos questions.")

    # Choix du framework
    framework = st.radio(
        "Choisissez le framework d'indexation",
        options=["langchain", "llamaindex"],
        index=0 if st.session_state['framework'] == "langchain" else 1
    )

    if framework != st.session_state['framework']:
        st.session_state['stored_files'] = []
        clear_indexes()
        st.session_state['framework'] = framework
        st.experimental_rerun()

    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
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
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                if framework == "langchain":
                    my_langchain.store_pdf_file(path, f.name)
                else:
                    my_llamaindex.store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    # Suppression des fichiers supprimés
    current_files = {f['Nom du fichier'] for f in file_info}
    files_to_be_deleted = set(st.session_state['stored_files']) - current_files
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        if framework == "langchain":
            my_langchain.delete_file_from_store(name)
        # Note : llamaindex ne gère pas la suppression d’un document individuellement

    # Slider k (pour Langchain uniquement pour l’instant)
    k = st.slider(
        label="Nombre de documents similaires à récupérer (k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Choix de langue (pas utilisé dans my_llamaindex.py, ok)
    language = st.selectbox(
        "Choisissez la langue de réponse",
        options=["français", "anglais", "espagnol", "allemand"],
        index=0
    )

    # Saisie de la question
    question = st.text_input("Votre question ici")

    if st.button("Analyser"):
        if not question:
            st.warning("Veuillez entrer une question avant d'analyser.")
        else:
            if framework == "langchain":
                model_response = my_langchain.answer_question(question, language, k)
            else:
                model_response = my_llamaindex.answer_question(question)
            st.text_area("Zone de texte, réponse du modèle", value=model_response, height=200)

            feedback = st.radio(
                "Que pensez-vous de la qualité de la réponse ?",
                options=[1, 2, 3, 4, 5],
                index=4,
                format_func=lambda x: f"{x} étoiles",
                key="user_feedback"
            )
            if feedback is not None:
                print(f"Feedback utilisateur: {feedback}")


if __name__ == "__main__":
    main()
