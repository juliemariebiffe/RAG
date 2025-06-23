import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleVectorStore,
    ServiceContext,
    Document,
    TextNode,
    VectorStoreQuery,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import AzureOpenAIEmbedding
from llama_index.llms.openai import AzureOpenAI
import fitz  # PyMuPDF

import pandas as pd

from utils import get_pdf_text

def process_files_with_llama_index(uploaded_files):
    pdf_text = get_pdf_text(uploaded_files)

    llm = AzureOpenAI(
        model="gpt-35-turbo",
        deployment_name=st.secrets["chat"]["azure_gpt_deployment"],
        api_key=st.secrets["chat"]["azure_api_key"],
        azure_endpoint=st.secrets["chat"]["azure_endpoint"],
        api_version=st.secrets["chat"]["azure_api_version"],
    )
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=st.secrets["embedding"]["azure_embedding_deployment"],
        api_key=st.secrets["embedding"]["azure_api_key"],
        azure_endpoint=st.secrets["embedding"]["azure_endpoint"],
        api_version=st.secrets["embedding"]["azure_api_version"],
    )

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

    documents = [Document(text=pdf_text)]
    splitter = SentenceSplitter()
    nodes = splitter.get_nodes_from_documents(documents)

    vector_store = SimpleVectorStore()
    index = VectorStoreIndex(nodes, service_context=service_context, vector_store=vector_store)

    return index, nodes

def ask_llama_index(index, query, top_k):
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return str(response)
