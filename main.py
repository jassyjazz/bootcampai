import openai
import json
import streamlit as st
import os
import requests
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain.agents import Tool
from langchain.runnables import RunnablePassthrough
from datetime import datetime
import pytz

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

# Password protection for the main page
def password_protection():
    password = st.text_input("Enter Password", type="password")
    if password != "bootcamp123" and password != "":
        st.error("Incorrect password. Please try again.")
        st.stop()

# Home Page with Password Protection
if "main" in st.session_state:
    password_protection()

# Function to retrieve resale flats based on the selected budget
url = f"https://data.gov.sg/api/action/datastore_search?resource_id=d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

def get_resale_flats_by_budget(budget, town, flat_type):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['result']['records']
        filtered_flats = []
        
        # Ensure budget is an integer
        budget = int(budget)
        
        for flat in data:
            try:
                # Convert resale_price to integer for comparison
                resale_price = int(flat['resale_price'])
                
                # Apply filters based on user's criteria
                if resale_price <= budget and (town == "Any" or flat['town'] == town) and (flat_type == "Any" or flat['flat_type'] == flat_type):
                    filtered_flats.append(flat)
            except ValueError:
                continue  # Skip flats with invalid resale_price
        
        # Display data in a table with formatted currency
        if filtered_flats:
            df = pd.DataFrame(filtered_flats, columns=["town", "flat_type", "resale_price", "storey_range", "floor_area_sqm"])
            df["resale_price"] = df["resale_price"].apply(lambda x: f"${int(x):,}")
            st.dataframe(df, width=1000)  # Set a larger table width
        else:
            st.write("No flats found matching your criteria.")  # This message will only show when no results are found

    except Exception as e:
        st.error(f"Error fetching flats data: {str(e)}")

# Page navigation logic
page = st.sidebar.selectbox("Choose a page", ["Home", "HDB resale chatbot", "HDB resale flat search", "About Us", "Methodology"])

# Home Page
if page == "Home":
    st.title("Welcome to the HDB Resale Flat Guide")
    st.markdown("""
    Welcome to our HDB Resale Flat Guide! This app helps you navigate the resale flat buying process in Singapore.
    You can learn about the buying procedure, interact with a virtual assistant, and search for available flats
    based on your budget.
    """)
    st.markdown("""
    To get started, you can explore the following pages:
    - **HDB Resale Chatbot**: Chat with our AI assistant about the HDB resale process.
    - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.
    - **About Us**: Learn more about our project, objectives, and data sources.
    - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
    """)

# HDB Resale Chatbot Page
elif page == "HDB resale chatbot":
    st.title("Hi! I am your virtual HDB assistant.")
    st.markdown("""
    I am here to help you understand the HDB resale process in Singapore. Feel free to ask me anything about buying
    an HDB resale flat, and I will provide you with all the relevant information.
    """)
    
    user_question = st.text_input("Ask me anything about the HDB resale process:")
    
    if user_question:
        response = chain.invoke(user_question)
        st.write(response)

    st.markdown("""
    I can guide you through the entire resale process, from understanding eligibility criteria to finding the right flat.
    Ask away!
    """)

    # Submit button
    st.button("Submit")

# HDB Resale Flat Search Page
elif page == "HDB resale flat search":
    st.title("Find Resale Flats Within Your Budget")

    st.markdown("""
    This page allows you to search for available HDB resale flats based on your budget. Use the filters below to refine your search:
    """)
    
    budget = st.slider("Select your budget", 100000, 10000000, 1000000, step=10000, format="₹{:,.0f}")
    town = st.selectbox("Select Town", ["Any"] + ["Town1", "Town2", "Town3"])  # Replace with actual towns
    flat_type = st.selectbox("Select Flat Type", ["Any", "1-Room", "2-Room", "3-Room", "4-Room", "5-Room", "Executive"])  # Example flat types

    get_resale_flats_by_budget(budget, town, flat_type)

# About Us Page
elif page == "About Us":
    st.title("About Us")
    st.markdown("""
    **Project Scope**: This project aims to provide a comprehensive guide to buying an HDB resale flat in Singapore. 
    It includes interactive tools such as a chatbot and flat search feature, and utilizes real-time data from data.gov.sg.
    
    **Objectives**: 
    - To educate users about the HDB resale process
    - To provide personalized flat search options based on user budget and preferences
    - To help users make informed decisions about buying a resale flat
    
    **Data Sources**: 
    - Data from data.gov.sg for resale flat listings
    - HDB website for the resale process and eligibility criteria
    """)

# Methodology Page
elif page == "Methodology":
    st.title("Methodology")
    st.markdown("""
    **Data Flow**: The application relies on multiple data sources, including the HDB website and data.gov.sg API.
    We fetch real-time resale flat data and integrate it with other features like the chatbot to provide comprehensive guidance.
    
    **Implementation Details**: 
    - The chatbot uses OpenAI's GPT-4 model for answering user queries.
    - The flat search is based on filtering data from the data.gov.sg API.
    
    **Flowchart**: 
    [Flowchart here depicting the use case logic]
    """)

