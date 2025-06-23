import os
import tempfile

import streamlit as st
import pandas as pd

from rag import my_langchain
from rag.my_llamaindex import MyLlamaIndex  # ✅ Nouvelle classe

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# Initialisation des états
if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []
if 'framework' not in st.session_state:
    st.session_state['framework'] = "langchain"
if 'llamaindex_instance' not in st.session_state:
    st.session_state['llamaindex_instance'] = MyLlamaIndex()
if 'file_buffers' not in st.session_state:
    st.session_state['file_buffers'] = {}  # Pour stocker les fichiers PDF en mémoire

llamaindex = st.session_state['llamaindex_instance']


def clear_indexes():
    my_langchain.clear_index()
    # Si tu veux, tu peux ajouter une fonction clear_index dans my_llamaindex.py
    # Ici on laisse vide ou on peut réinitialiser le vector_store si tu modifies my_llamaindex.py


def rebuild_langchain_index():
    """Reconstruit l'index Langchain à partir des fichiers en mémoire"""
    my_langchain.clear_index()
    for filename, file_bytes in st.session_state['file_buffers'].items():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            my_langchain.store_pdf_file(tmp_file.name, filename)
            # Suppression du fichier temporaire après stockage
            os.unlink(tmp_file.name)


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
        st.session_state['file_buffers'] = {}
        clear_indexes()
        st.session_state['framework'] = framework
        st.experimental_rerun()

    uploaded_files = st.file_uploader(
        label="Déposez vos fichiers ici ou chargez-les",
        accept_multiple_files=True,
        type=["pdf"]
    )

    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_in_kb:.2f}"
            })

            # Stockage en mémoire si nouveau fichier
            if f.name not in st.session_state['file_buffers']:
                st.session_state['file_buffers'][f.name] = f.getvalue()
                st.session_state['stored_files'].append(f.name)

        # Si on utilise langchain, on reconstruit l'index à partir des fichiers en mémoire
        if framework == "langchain":
            rebuild_langchain_index()
        else:
            # Pour llamaindex, on stocke juste les nouveaux fichiers
            for f in uploaded_files:
                if f.name not in st.session_state['stored_files']:
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as outfile:
                        outfile.write(f.read())
                    llamaindex.store_pdf_file(path, f.name)

    else:
        # Si aucun fichier uploadé, on affiche la liste actuelle en mémoire
        for name in st.session_state['stored_files']:
            # Taille inconnue ici car pas accessible directement, on peut afficher "-"
            file_info.append({"Nom du fichier": name, "Taille (KB)": "-"})

    # Gestion suppression fichiers (si utilisateur supprime via uploader)
    current_files = {f['Nom du fichier'] for f in file_info}
    files_to_be_deleted = set(st.session_state['stored_files']) - current_files
    if files_to_be_deleted:
        for name in files_to_be_deleted:
            st.session_state['stored_files'].remove(name)
            if name in st.session_state['file_buffers']:
                del st.session_state['file_buffers'][name]

        if framework == "langchain":
            rebuild_langchain_index()
        else:
            # llamaindex ne gère pas la suppression individuelle
            pass

    # Affichage tableau fichiers
    if file_info:
        df = pd.DataFrame(file_info)
        st.table(df)

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
                model_response = llamaindex.answer_question(question)
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
