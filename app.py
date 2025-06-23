import os
import tempfile

import streamlit as st
import pandas as pd

from rag.langchain import answer_question
from rag.langchain import delete_file_from_store
from rag.langchain import store_pdf_file

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="👋",
)

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
                path = os.path.join(temp_dir, "temp.pdf")  # Give it a name, like temp.pdf
                with open(path, "wb") as outfile:
                    outfile.write(f.read()) # Use f.read() to get the bytes
                store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)
        


        df = pd.DataFrame(file_info)
        st.table(df)  # on affiche le tableau

    # Gestion de la suppression de documents
    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    # print(set(st.session_state['stored_files']))
    # print({f['Nom du fichier'] for f in file_info})
    # print(files_to_be_deleted)
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        delete_file_from_store(name)

    # Champ de question
    question = st.text_input("Votre question ici")

    # Bouton pour lancer l’analyse
    if st.button("Analyser"):
        # ========
        # ICI : vous pouvez implémenter la logique d’analyse,
        # par exemple interroger un modèle de NLP (ex: GPT, spaCy, etc.)
        # en lui passant la question et le contenu des fichiers.
        # ========
        
        # On met un placeholder de réponse pour la démonstration
        #reponse_modele = f"Voici une réponse fictive à la question : {question}"
        
        model_response = answer_question(question)

        # Affichage de la réponse
        st.text_area("Zone de texte, réponse du modèle",
                     value=model_response, height=200)
    else:
        # Zone vide ou explicative quand on n'a pas encore analysé
        st.text_area("Zone de texte, réponse du modèle", value="", height=200)

if __name__ == "__main__":
    main()
