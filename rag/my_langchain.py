import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI


CHUNK_SIZE = 1_000
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

def store_pdf_file(path: str, filename: str):
    # Implémentation pour stocker et vectoriser le PDF
    config = get_embedding_config()
    # Utilise config["api_key"], etc. pour appeler Azure ou autre
    # Par exemple, créer un vecteur à partir du pdf et stocker dans un vecteur store
    print(f"Stockage de {filename} avec clé {config['api_key'][:4]}...")  # Debug

def delete_file_from_store(filename: str):
    # Implémentation pour supprimer un document du store
    print(f"Suppression de {filename} du store")

def answer_question(question: str) -> str:
    config = get_chat_config()
    # Appelle l'API OpenAI Azure ou autre pour répondre à la question
    # Ici placeholder
    print(f"Question posée : {question}, avec clé {config['api_key'][:4]}...")
    return f"Réponse fictive à la question : {question}"



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

def get_meta_doc(extract: str) -> str:
    """Generate a synthetic metadata description of the content.
    """
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
        - type of content (e.g. scientific paper, litterature, news, etc.)
        - language
        - themes as a list of keywords

        <content>
        {}
        </content>
        """.format(extract),
    ),]
    response = llm.invoke(messages)
    return response.content


def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    """Store a pdf file in the vector store.

    Args:
        file_path (str): file path to the PDF file
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    # TODO: make a constant of chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)
    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
            }
    if use_meta_doc:
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(page_content=get_meta_doc(extract),
                            metadata={
                                'document_name': doc_name,
                                'insert_date': datetime.now()
                                })
        all_splits.append(meta_doc)
    _ = vector_store.add_documents(documents=all_splits)
    return


def delete_file_from_store(name: str) -> int:
    ids_to_remove = []
    for (id, doc) in vector_store.store.items():
        if name == doc['metadata']['document_name']:
            ids_to_remove.append(id)
    vector_store.delete(ids_to_remove)
    #print('File deleted:', name)
    return len(ids_to_remove)


def inspect_vector_store(top_n: int=10) -> list:
    docs = []
    for index, (id, doc) in enumerate(vector_store.store.items()):
        if index < top_n:
            docs.append({
                'id': id,
                'document_name': doc['metadata']['document_name'],
                'insert_date': doc['metadata']['insert_date'],
                'text': doc['text']
                })
            # docs have keys 'id', 'vector', 'text', 'metadata'
            # print(f"{id} {doc['metadata']['document_name']}: {doc['text']}")
        else:
            break
    return docs


def get_vector_store_info():
    nb_docs = 0
    max_date, min_date = None, None
    documents = set()
    for (id, doc) in vector_store.store.items():
        nb_docs += 1
        if max_date is None or max_date < doc['metadata']['insert_date']:
            max_date = doc['metadata']['insert_date']
        if min_date is None or min_date > doc['metadata']['insert_date']:
            min_date = doc['metadata']['insert_date']
        documents.add(doc['metadata']['document_name'])
    return {
        'nb_chunks': nb_docs,
        'min_insert_date': min_date,
        'max_insert_date': max_date,
        'nb_documents': len(documents)
    }


def retrieve(question: str):
    """Retrieve documents similar to a question.

    Args:
        question (str): text of the question

    Returns:
        list[TODO]: list of similar documents retrieved from the vector store
    """
    retrieved_docs = vector_store.similarity_search(question)
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


def answer_question(question: str, language: str = "français") -> str:
    """Answer a question by retrieving similar documents in the store.

    Args:
        question (str): text of the question

    Returns:
        str: text of the answer
    """
    inspect_vector_store()
    docs = retrieve(question)
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

