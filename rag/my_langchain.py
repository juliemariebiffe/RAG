import streamlit as st
from datetime import datetime

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import InMemoryVectorStore
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

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"]
)

def clear_index():
    global vector_store
    vector_store = InMemoryVectorStore(embedder)

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
    for id in ids_to_remove:
        vector_store.store.pop(id, None)
    return len(ids_to_remove)

def answer_question(question: str, language: str = "français", k: int = 5) -> str:
    global vector_store, llm

    if not vector_store or len(vector_store.store) == 0:
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
