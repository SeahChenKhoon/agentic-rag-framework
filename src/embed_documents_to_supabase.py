import os
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# import utils
import src.utils.config_loader as config_loader

# load environment variables
load_dotenv()  

config_file_path = os.environ.get("CONFIG_FILE_PATH")
config = config_loader.load_config(config_file_path)

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model=config["embedding"]["model"])

# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader(config["paths"]["documents"])

# split the documents in multiple chunks
documents = loader.load()
chunk_size=config["embedding"]["chunk"]["size"]
chunk_overlap=config["embedding"]["chunk"]["overlap"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
docs = text_splitter.split_documents(documents)

table_name=config["database"]["supabase"]["table"]
query_name=config["database"]["supabase"]["query_function"]

# store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name=table_name,
    query_name=query_name,
    chunk_size=chunk_size,
)
