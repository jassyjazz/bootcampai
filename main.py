import openai
import json
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough

# Check if the OpenAI API key is set in Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets["general"]:
    st.error("OpenAI API key not found in Streamlit secrets. Please set the OPENAI_API_KEY in the app's secrets.")
    st.stop()

# Set OpenAI API key from Streamlit secrets
api_key = st.secrets["general"]["OPENAI_API_KEY"]
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

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
    try:
        documents = load_document()
        relevant_info = ""
        for section in documents.get('sections', []):
            if any(keyword in user_prompt.lower() for keyword in section.get('keywords', [])):
                relevant_info += f"{section['content']}\n\n"
        return relevant_info or "No relevant information found."
    except Exception as e:
        st.error(f"Error in document retrieval: {str(e)}")
        return "Error occurred while retrieving documents."

# Define tools for the LLM agent
tools = [
    Tool(
        name="Document Retrieval",
        func=retrieve_relevant_documents,
        description="Useful for retrieving relevant information about HDB resale process"
    )
]

# Define prompt template for the agent
prompt_template = PromptTemplate(
    input_variables=["relevant_docs", "question"],
    template='You are an AI assistant guiding users through the HDB resale process in Singapore. '
             'Use the following information to answer the user\'s question:\n\n'
             '{relevant_docs}\n\n'
             'Human: {question}\n'
             'AI: '
)

# Initialize the LLM with explicit API key
llm = OpenAI(temperature=0, openai_api_key=api_key)

# Create the chain
chain = (
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Streamlit interface
st.title("HDB Resale Guide")

user_question = st.text_input("Ask a question about the HDB resale process:")

if user_question:
    response = chain.invoke(user_question)
    st.write(response)
