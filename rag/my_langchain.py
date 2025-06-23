import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

config = st.secrets

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    openai_api_version=config["embedding"]["azure_api_version"],
    api_key=config["embedding"]["azure_api_key"]
)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"]
)

vector_store = None

def clear_index():
    global vector_store
    vector_store = None

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    global vector_store

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
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={
                'document_name': doc_name,
                'insert_date': datetime.now()
            }
        )
        all_splits.append(meta_doc)

    if vector_store is None:
        vector_store = FAISS.from_documents(all_splits, embedder)
    else:
        vector_store.add_documents(all_splits)

def delete_file_from_store(doc_name: str) -> int:
    global vector_store
    clear_index()
    return 0

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    global vector_store, llm

    if vector_store is None or len(vector_store.index_to_docstore_id) == 0:
        return "Aucun document indexé, veuillez charger des documents."

    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an assistant answering questions in {language}. "
                "Use the provided context to answer the question. "
                "If unsure, say you don't know."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

    response = llm.invoke(messages)
    return response.content

def get_meta_doc(extract: str) -> str:
    return f"Résumé automatique : {extract}"
