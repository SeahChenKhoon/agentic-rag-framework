# import basics
import os
from dotenv import load_dotenv
from typing import Union
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_openai import AzureOpenAIEmbeddings

from supabase.client import Client, create_client
from langchain_core.tools import tool

# import utils
import src.utils.config_loader as config_loader

# load environment variables
load_dotenv()  

config_file_path = os.environ.get("CONFIG_FILE_PATH")
config = config_loader.load_config(config_file_path)

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings(
    model=config["embedding"]["model"]
)

table_name=config["database"]["supabase"]["table"]
query_name=config["database"]["supabase"]["query_function"]

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name=table_name,
    query_name=query_name,
)

# initiate large language model (temperature = 0)
llm = ChatOpenAI(temperature=config["llm"]["temperature"])


# Convert YAML prompt config to LangChain message templates
messages = []
for entry in config["prompt"]:
    if entry["type"] == "placeholder":
        messages.append(
            MessagesPlaceholder(
                variable_name=entry["variable_name"],
                optional=entry.get("optional", False)
            )
        )
    else:
        messages.append((entry["type"], entry["content"]))

# Build ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(messages)

# create the tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# invoke the agent
response = agent_executor.invoke({"input": config["query"]["default"]})

# put the result on the screen
print(response["output"])