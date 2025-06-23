import os
import tempfile

import streamlit as st
import pandas as pd

from rag import my_langchain
from rag.my_llamaindex import MyLlamaIndex  # ✅ Nouvelle classe LlamaIndex

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

# Initialisation des états Streamlit pour garder la trace des fichiers, framework et index
if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []  # Liste des noms de fichiers chargés
if 'framework' not in st.session_state:
    st.session_state['framework'] = "langchain"  # Framework utilisé (langchain ou llamaindex)
if 'llamaindex_instance' not in st.session_state:
    st.session_state['llamaindex_instance'] = MyLlamaIndex()  # Instance singleton LlamaIndex
if 'file_buffers' not in st.session_state:
    st.session_state['file_buffers'] = {}  # Pour stocker en mémoire les fichiers PDF (LangChain)

llamaindex = st.session_state['llamaindex_instance']


def clear_indexes():
    """
    Réinitialise les indexes de LangChain et LlamaIndex.
    Important lors du changement de framework ou de suppression totale.
    """
    my_langchain.clear_index()
    llamaindex.clear_index()


def rebuild_langchain_index():
    """
    Reconstruit complètement l'index LangChain à partir des fichiers PDF
    stockés en mémoire dans `file_buffers`.
    Utilisé après ajout ou suppression de fichiers en mode LangChain.
    """
    my_langchain.clear_index()
    for filename, file_bytes in st.session_state['file_buffers'].items():
        # Création d'un fichier temporaire pour que LangChain puisse le charger
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()
            # Stockage dans l'index LangChain
            my_langchain.store_pdf_file(tmp_file.name, filename)
            # Suppression du fichier temporaire immédiatement après l'indexation
            os.unlink(tmp_file.name)


def main():
    st.title("Analyse de documents")
    st.subheader("Chargez vos documents et posez vos questions à l’IA")

    # Sélection du framework d'indexation à utiliser
    framework = st.radio(
        "Choisissez le framework d'indexation",
        options=["langchain", "llamaindex"],
        index=0 if st.session_state['framework'] == "langchain" else 1
    )

    # Si changement de framework, on reset les états, fichiers et indexes
    if framework != st.session_state['framework']:
        st.session_state['stored_files'] = []
        st.session_state['file_buffers'] = {}
        clear_indexes()
        st.session_state['framework'] = framework
        st.experimental_rerun()  # Recharge la page pour appliquer les changements

    # Upload multiple fichiers PDF
    uploaded_files = st.file_uploader(
        "Déposez vos fichiers ici ou chargez-les",
        accept_multiple_files=True,
        type=["pdf"]
    )

    file_info = []  # Pour afficher le tableau des fichiers chargés

    if uploaded_files:
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_kb:.2f}"
            })

            # Ajout uniquement si le fichier n'a pas déjà été chargé
            if f.name not in st.session_state['stored_files']:
                st.session_state['stored_files'].append(f.name)

                if framework == "langchain":
                    # Pour LangChain, on stocke le contenu en mémoire
                    st.session_state['file_buffers'][f.name] = f.getvalue()
                else:
                    # Pour LlamaIndex, on écrit le fichier dans un répertoire temporaire
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, f.name)
                    with open(temp_path, "wb") as tmpf:
                        tmpf.write(f.getvalue())
                    # On indexe immédiatement ce fichier
                    llamaindex.store_pdf_file(temp_path, f.name)

        # Reconstruction de l'index LangChain uniquement (car stock en mémoire)
        if framework == "langchain":
            rebuild_langchain_index()

    else:
        # Aucun fichier uploadé, affichage de la liste des fichiers déjà stockés
        for name in st.session_state['stored_files']:
            # Taille inconnue ici car non accessible, affichage "-"
            file_info.append({"Nom du fichier": name, "Taille (KB)": "-"})

    # Gestion des fichiers supprimés par l'utilisateur via l'interface uploader
    current_files = {f['Nom du fichier'] for f in file_info}
    removed_files = set(st.session_state['stored_files']) - current_files
    if removed_files:
        for name in removed_files:
            st.session_state['stored_files'].remove(name)
            # Suppression du buffer en mémoire uniquement pour LangChain
            if framework == "langchain" and name in st.session_state['file_buffers']:
                del st.session_state['file_buffers'][name]

        # Reconstruction ou nettoyage des index en fonction du framework
        if framework == "langchain":
            rebuild_langchain_index()
        else:
            # LlamaIndex ne gère pas la suppression individuelle proprement
            # Il faudrait envisager de reconstruire l'index complet si suppression
            pass

    # Affichage du tableau des fichiers chargés avec leur taille
    if file_info:
        df = pd.DataFrame(file_info)
        st.table(df)

    # Slider pour choisir le nombre de documents similaires à récupérer (k) - LangChain uniquement
    k = st.slider(
        "Nombre de documents similaires à récupérer (k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Choix de la langue pour la réponse (utilisé uniquement pour LangChain)
    language = st.selectbox(
        "Choisissez la langue de réponse",
        options=["français", "anglais", "espagnol", "allemand"],
        index=0
    )

    # Champ texte pour poser une question
    question = st.text_input("Votre question ici")

    # Bouton d'analyse
    if st.button("Analyser"):
        if not question:
            st.warning("Veuillez entrer une question avant d'analyser.")
        else:
            # Appel de la fonction de réponse selon le framework choisi
            if framework == "langchain":
                model_response = my_langchain.answer_question(question, language, k)
            else:
                model_response = llamaindex.answer_question(question)

            # Affichage de la réponse dans une zone texte
            st.text_area("Réponse du modèle", value=model_response, height=200)

            # Système simple de feedback utilisateur (1 à 5 étoiles)
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
