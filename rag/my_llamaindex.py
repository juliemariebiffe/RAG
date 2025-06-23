from llama_index import VectorStoreIndex
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class MyLlamaIndex:
    def __init__(self):
        self.index = None
        self.documents = []

    def clear_index(self):
        self.index = None
        self.documents = []

    def store_pdf_file(self, file_path: str, doc_name: str):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)

        for split in splits:
            split.metadata = {
                "document_name": doc_name,
                "insert_date": datetime.now()
            }

        self.documents.extend(splits)
        self._build_index()

    def _build_index(self):
        if not self.documents:
            self.index = None
            return
        self.index = VectorStoreIndex.from_documents(self.documents)

    def answer_question(self, question: str) -> str:
        if self.index is None:
            return "Aucun document indexé, veuillez charger des documents."
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return str(response)
