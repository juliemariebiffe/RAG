import os
from datetime import datetime
import streamlit as st

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI



class MyLlamaIndex:
    def __init__(self):
        config = st.secrets

        # Initialisation des embeddings et du modèle
        self.embed_model = resolve_embed_model("local")

        self.llm = ChatOpenAI(
            openai_api_key=config["chat"]["azure_api_key"],
            openai_api_base=config["chat"]["azure_endpoint"],
            openai_api_version=config["chat"]["azure_api_version"],
            deployment_name=config["chat"]["azure_deployment"],
            model="gpt-35-turbo",  # ou gpt-4 selon ton usage
        )

        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )

        self.documents = []
        self.index = None
        self.chat_engine = None

    def store_pdf_file(self, file_path: str, doc_name: str):
        reader = SimpleDirectoryReader(input_files=[file_path])
        new_docs = reader.load_data()

        for doc in new_docs:
            doc.metadata["name"] = doc_name
            doc.metadata["insert_date"] = str(datetime.now())

        self.documents.extend(new_docs)
        self._update_index()

    def _update_index(self):
        if self.documents:
            self.index = VectorStoreIndex.from_documents(
                self.documents, service_context=self.service_context
            )
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True
            )

    def answer_question(self, question: str) -> str:
        if not self.chat_engine:
            return "Aucun document indexé, veuillez charger des documents."

        response = self.chat_engine.chat(question)
        return str(response)

    def clear_index(self):
        self.documents = []
        self.index = None
        self.chat_engine = None
