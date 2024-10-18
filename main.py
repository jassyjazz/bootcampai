import openai
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory

# Load the document for RAG from JSON
def load_document():
    try:
        with open("docs/buy_hdb_resale.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Error: 'buy_hdb_resale.json' file not found. Check the file path.")
        raise
    except json.JSONDecodeError:
        st.error("Error: 'buy_hdb_resale.json' contains invalid JSON. Please check the format.")
        raise

# Function to retrieve relevant documents
def retrieve_relevant_documents(user_prompt):
    documents = load_document()
    relevant_info = ""
    for section in documents.get('sections', []):
        if any(keyword in user_prompt.lower() for keyword in section.get('keywords', [])):
            relevant_info += f"{section['content']}\n\n"
    return relevant_info or "No relevant information found."

# Define tools for the LLM agent
tools = [
    Tool(
        name="Document Retrieval",
        func=retrieve_relevant_documents,
        description="Useful for retrieving relevant information about HDB resale process"
    )
]

# Define prompt template for the agent
prompt_template = """You are an AI assistant guiding users through the HDB resale process in Singapore. 
Use the following information to answer the user's question:

{relevant_docs}
