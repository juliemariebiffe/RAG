import streamlit as st
from datetime import datetime

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores import VectorStoreQuery

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

# Récupération des secrets depuis streamlit secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Initialisation embeddings OpenAI
embedder = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-3-large",  # adapte selon ton besoin
)

# Initialisation modèle ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",  # adapte selon ton besoin
    temperature=0.0,
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()


def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embedder.embed_query(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    vector_store.add(nodes)


def delete_file_from_store(name: str) -> int:
    raise NotImplemented('function not implemented for Llamaindex')


def inspect_vector_store(top_n: int = 10) -> list:
    raise NotImplemented('function not implemented for Llamaindex')


def get_vector_store_info():
    raise NotImplemented('function not implemented for Llamaindex')


def retrieve(question: str):
    query_embedding = embedder.embed_query(question)

    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
    )

    query_result = vector_store.query(vector_store_query)
    return query_result.nodes


def build_qa_messages(question: str, context: str) -> list[tuple[str, str]]:
    messages = [
        (
            "system",
            "You are an assistant for question-answering tasks.",
        ),
        (
            "system",
            f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{context}""",
        ),
        (
            "user",
            question,
        ),
    ]
    return messages


def answer_question(question: str) -> str:
    docs = retrieve(question)
    docs_content = "\n\n".join(doc.get_content() for doc in docs)

    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")

    messages = build_qa_messages(question, docs_content)
    response = llm.invoke(messages)
    return response.content
