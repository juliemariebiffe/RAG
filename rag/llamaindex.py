import yaml
from datetime import datetime

from llama_index import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.readers.file.base import DEFAULT_PDF_READER
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser

# Chargement config
def read_config(file_path="secrets/config.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

config = read_config()

# Setup LLM et embedder avec llama_index
llm = OpenAI(
    model=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_openai=True,
    deployment_name=config["chat"]["azure_deployment"],
    azure_api_version=config["chat"]["azure_api_version"],
    azure_endpoint=config["chat"]["azure_endpoint"],
)

embedder = OpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_openai=True,
    deployment_name=config["embedding"]["azure_deployment"],
    azure_api_version=config["embedding"]["azure_api_version"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedder)

index = None  # Index global, à charger ou créer

def store_pdf_file(file_path: str, doc_name: str):
    global index
    loader = DEFAULT_PDF_READER(file_path)
    documents = loader.load_data()

    # Création ou mise à jour de l'index
    if index is None:
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    else:
        index.insert(documents)
    # Optionnel: sauvegarde de l'index dans un dossier local

def answer_question(question: str) -> str:
    global index
    if index is None:
        return "Aucun document indexé. Veuillez charger des documents d'abord."
    response = index.query(question)
    return response.response

def delete_file_from_store(name: str):
    raise NotImplementedError("Suppression non implémentée pour LlamaIndex")

