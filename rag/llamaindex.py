import streamlit as st
from datetime import datetime

from llama_index import (
    GPTVectorStoreIndex,
    SimpleVectorStore,
    ServiceContext,
    TextNode,
)
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index.node_parser import SentenceSplitter
from llama_index.indices.query.vector_store.base import VectorStoreQuery

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Récupération directe des secrets Streamlit
config = st.secrets  # dictionnaire des secrets définis dans secrets.toml

# Initialisation LLM et embedder Azure OpenAI
llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

# Créer un contexte de service regroupant LLM et embedding (recommandé)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedder)

# Index global LlamaIndex, None si non créé ou réinitialisé
index = None
vector_store = SimpleVectorStore()

def clear_index():
    global index, vector_store
    index = None
    vector_store = SimpleVectorStore()  # Remise à zéro du store

def store_pdf_file(file_path: str, doc_name: str):
    global index, vector_store
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

    # Calculer les embeddings pour chaque noeud
    for node in nodes:
        node.embedding = embedder.get_text_embedding(
            node.get_content(metadata_mode="all")
        )

    vector_store.add(nodes)

    # Construire ou reconstruire l’index avec le vecteur mis à jour
    index = GPTVectorStoreIndex(vector_store=vector_store, service_context=service_context)

def delete_file_from_store(name: str) -> int:
    raise NotImplementedError('Suppression non implémentée pour LlamaIndex')

def inspect_vector_store(top_n: int=10) -> list:
    raise NotImplementedError('Inspection non implémentée pour LlamaIndex')

def get_vector_store_info():
    raise NotImplementedError('Info store non implémentée pour LlamaIndex')

def retrieve(question: str):
    global index
    if index is None:
        return []
    query_embedding = embedder.get_query_embedding(question)
    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
    )
    query_result = index.query(vector_store_query)
    return query_result.nodes

def build_qa_messages(question: str, context: str) -> list[tuple[str, str]]:
    messages = [
        (
            "system",
            "You are an assistant for question-answering tasks.",
        ),
        (
            "system",
            f"Use the following pieces of retrieved context to answer the question.\n"
            f"If you don't know the answer, just say that you don't know.\n"
            f"Use three sentences maximum and keep the answer concise.\n"
            f"{context}"
        ),
        (
            "user",
            question
        ),
    ]
    return messages

def answer_question(question: str) -> str:
    global index
    docs = retrieve(question)
    if not docs:
        return "Aucun document indexé, veuillez charger des documents."
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", getattr(doc, 'id', 'no-id'))
        print(doc.get_content())
        print("------")
    messages = build_qa_messages(question, docs_content)
    response = llm.invoke(messages)
    return response.content
