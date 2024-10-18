import openai
import json
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough

# Scrape HDB website and cache data for 24 hours
@st.cache_data(ttl=86400)
def scrape_hdb_resale_info():
    url = "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/buying-procedure-for-resale-flats/overview"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant content, customize based on actual structure
        content_sections = soup.find_all('section')  # Adjust selector based on the page structure
        scraped_data = []
        for section in content_sections:
            title = section.find('h2').get_text(strip=True) if section.find('h2') else "No Title"
            content = section.get_text(strip=True)
            scraped_data.append({'title': title, 'content': content})
        return scraped_data
    except Exception as e:
        st.error(f"Error scraping HDB website: {str(e)}")
        return []

# Load the fallback document for RAG from JSON
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

# Retrieve relevant information from scraped data or fallback to JSON
def retrieve_relevant_documents(user_prompt):
    try:
        scraped_data = scrape_hdb_resale_info()
        relevant_info = ""
        
        # Search the scraped data first
        for section in scraped_data:
            if user_prompt.lower() in section['content'].lower():
                relevant_info += f"{section['title']}: {section['content']}\n\n"
        
        # If no relevant info is found, fall back to the JSON document
        if not relevant_info:
            documents = load_document()
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

# Initialize the LLM with gpt-4o-mini chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Specify the gpt-4o-mini chat model here
    temperature=0, 
    openai_api_key=st.secrets["general"]["OPENAI_API_KEY"]
)

# Create the chain
chain = (
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Streamlit interface
st.title("HDB Resale Guide")

# Main introduction page
st.write("""
Welcome to the HDB Resale Guide! Here, you can find information on the process of buying an HDB resale flat.
Ask any questions you have about the process, and we'll guide you step by step.
""")

# User query input for the chatbot
user_question = st.text_input("Ask a question about the HDB resale process:")

if user_question:
    response = chain.invoke(user_question)
    st.write(response)

# Section: HDB Resale Flat Search based on Budget
st.header("HDB Resale Flat Search by Budget")

# Slider for budget selection
budget = st.slider("Select your budget (SGD):", min_value=100000, max_value=1500000, step=50000)

# Fetch data from data.gov.sg API based on budget
datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
url = f"https://data.gov.sg/api/action/datastore_search?resource_id={datasetId}&limit=100"

def get_resale_flats_by_budget(budget):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['result']['records']
        
        # Filter flats by budget
        filtered_flats = [flat for flat in data if int(flat['resale_price']) <= budget]
        
        if filtered_flats:
            for flat in filtered_flats:
                st.write(f"Town: {flat['town']}, Flat Type: {flat['flat_type']}, Price: {flat['resale_price']}")
        else:
            st.write("No flats found within this budget range.")
    except Exception as e:
        st.error(f"Error fetching resale flats: {str(e)}")

get_resale_flats_by_budget(budget)
