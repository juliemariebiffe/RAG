import streamlit as st
from llama_index import (
    GPTVectorStoreIndex,
    SimpleVectorStore,
    ServiceContext,
    TextNode,
    VectorStoreQuery,
)
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index.node_parser import SentenceSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

config = st.secrets

# Initialisation LLM et embedder Azure
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

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedder)

# Index global et vector store global
index = None
vector_store = SimpleVectorStore()

def clear_index():
    global index, vector_store
    index = None
    vector_store = SimpleVectorStore()

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

    # Calcul des embeddings et ajout au vector store
    for node in nodes:
        node.embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
    vector_store.add(nodes)

    # (Re)création de l’index avec le vector store mis à jour
    index = GPTVectorStoreIndex(vector_store=vector_store, service_context=service_context)

def retrieve(question: str, k: int = 5):
    global index
    if index is None:
        return []
    query_embedding = embedder.get_query_embedding(question)
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=k,
        mode="default",
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

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question, k)
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
