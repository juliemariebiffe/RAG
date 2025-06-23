import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

config = st.secrets

def get_embedding_config():
    return {
        "api_key": st.secrets["embedding"]["azure_api_key"],
        "endpoint": st.secrets["embedding"]["azure_endpoint"],
        "deployment": st.secrets["embedding"]["azure_deployment"],
        "api_version": st.secrets["embedding"]["azure_api_version"],
    }

def get_chat_config():
    return {
        "api_key": st.secrets["chat"]["azure_api_key"],
        "endpoint": st.secrets["chat"]["azure_endpoint"],
        "deployment": st.secrets["chat"]["azure_deployment"],
        "api_version": st.secrets["chat"]["azure_api_version"],
    }

# Index global (variable globale) pour stocker les vecteurs
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    openai_api_version=config["embedding"]["azure_api_version"],
    api_key=config["embedding"]["azure_api_key"]
)

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"]
)

def clear_index():
    global vector_store
    vector_store = InMemoryVectorStore(embedder)  # Remettre à zéro l'index

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }

    if use_meta_doc:
        # Extraction meta info sur premiers chunks pour résumé
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={
                'document_name': doc_name,
                'insert_date': datetime.now()
            }
        )
        all_splits.append(meta_doc)

    global vector_store
    vector_store.add_documents(documents=all_splits)

def delete_file_from_store(doc_name: str) -> int:
    global vector_store
    ids_to_remove = []
    for (id, doc) in vector_store.store.items():
        if doc.metadata.get('document_name', '') == doc_name:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    return len(ids_to_remove)

def inspect_vector_store(top_n: int=10) -> list:
    global vector_store
    docs = []
    for index, (id, doc) in enumerate(vector_store.store.items()):
        if index >= top_n:
            break
        docs.append({
            'id': id,
            'document_name': doc.metadata.get('document_name', ''),
            'insert_date': doc.metadata.get('insert_date', None),
            'text': doc.page_content
        })
    return docs

def get_vector_store_info():
    global vector_store
    nb_docs = 0
    max_date, min_date = None, None
    documents = set()
    for (id, doc) in vector_store.store.items():
        nb_docs += 1
        insert_date = doc.metadata.get('insert_date')
        if max_date is None or (insert_date and max_date < insert_date):
            max_date = insert_date
        if min_date is None or (insert_date and min_date > insert_date):
            min_date = insert_date
        documents.add(doc.metadata.get('document_name', ''))
    return {
        'nb_chunks': nb_docs,
        'min_insert_date': min_date,
        'max_insert_date': max_date,
        'nb_documents': len(documents)
    }

def retrieve(question: str, k: int = 5):
    global vector_store
    retrieved_docs = vector_store.similarity_search(question, k=k)
    return retrieved_docs

def build_qa_messages(question: str, context: str, language: str) -> list[str]:
    messages = [
        (
            "system",
            "You are an assistant for question-answering tasks.",
        ),
        (
            "system",
            f"Use the following pieces of retrieved context to answer the question in {language}. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            f"{context}"
        ),
        (
            "user",
            question
        ),
    ]
    return messages

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    global llm
    docs = retrieve(question, k)
    if not docs:
        return "Aucun document indexé, veuillez charger des documents."
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")
    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content

def get_meta_doc(extract: str) -> str:
    messages = [
        (
            "system",
            "You are a librarian extracting metadata from documents.",
        ),
        (
            "user",
            """Extract from the content the following metadata.
            Answer 'unknown' if you cannot find or generate the information.
            Metadata list:
            - title
            - author
            - source
            - type of content (e.g. scientific paper, literature, news, etc.)
            - language
            - themes as a list of keywords

            <content>
            {}
            </content>
            """.format(extract),
        ),
    ]
    response = llm.invoke(messages)
    return response.content
