import openai
import json
import streamlit as st
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough

# Function to scrape HDB website content
@st.cache_data(ttl=86400)
def scrape_hdb_resale_info():
    url = "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/buying-procedure-for-resale-flats/overview"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content_sections = soup.find_all('section')
        scraped_data = []
        for section in content_sections:
            title = section.find('h2').get_text(strip=True) if section.find('h2') else "No Title"
            content = section.get_text(strip=True)
            scraped_data.append({'title': title, 'content': content})
        return scraped_data
    except Exception as e:
        st.error(f"Error scraping HDB website: {str(e)}")
        return []

# Load fallback document for RAG from JSON
def load_document():
    try:
        with open("docs/buy_hdb_resale.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Error: 'buy_hdb_resale.json' file not found.")
        raise
    except json.JSONDecodeError:
        st.error("Error: 'buy_hdb_resale.json' contains invalid JSON.")
        raise

# Retrieve relevant documents
def retrieve_relevant_documents(user_prompt):
    try:
        scraped_data = scrape_hdb_resale_info()
        relevant_info = ""
        for section in scraped_data:
            if user_prompt.lower() in section['content'].lower():
                relevant_info += f"{section['title']}: {section['content']}\n\n"
        
        if not relevant_info:
            documents = load_document()
            for section in documents.get('sections', []):
                if any(keyword in user_prompt.lower() for keyword in section.get('keywords', [])):
                    relevant_info += f"{section['content']}\n\n"
        
        return relevant_info or "No relevant information found."
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return "Error occurred while retrieving documents."

# Initialize the LLM with gpt-4o-mini model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0, 
    openai_api_key=st.secrets["general"]["OPENAI_API_KEY"]
)

# Create the prompt chain
prompt_template = PromptTemplate(
    input_variables=["relevant_docs", "question"],
    template='You are an AI assistant guiding users through the HDB resale process in Singapore. '
             'Use the following information to answer the user\'s question:\n\n'
             '{relevant_docs}\n\n'
             'Human: {question}\n'
             'AI: '
)

chain = (
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough() }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Streamlit App Pages

# Password entry and validation
password = st.text_input("Enter password to access the main page:", type="password")

# Check if the entered password is correct
if password != "bootcamp123" and password != "":
    st.error("Incorrect password. Please try again.")
    st.stop()  # Stop execution if password is incorrect
elif password == "":
    st.write("Please enter your password to proceed.")
    st.stop()  # Stops execution and asks user for password

# The rest of the app logic starts only if the password is correct
page = st.sidebar.selectbox("Select a page", ["Home", "About Us", "Methodology", "HDB Resale Chatbot", "HDB Resale Flat Search"])

# Dynamic title based on the selected page
if page == "Home":
    st.title("Welcome to the HDB Resale Guide!")
elif page == "About Us":
    st.title("About Us - HDB Resale Guide")
elif page == "Methodology":
    st.title("Methodology - HDB Resale Guide")
elif page == "HDB Resale Chatbot":
    st.title("HDB Resale Chatbot")
elif page == "HDB Resale Flat Search":
    st.title("HDB Resale Flat Search")

# Handle content for each page
if page == "Home":
    st.header("Welcome to the HDB Resale Guide!")
    st.write("""
    This application is designed to help you navigate the process of buying an HDB flat in the resale market.
    You can learn about the buying procedure, interact with a virtual assistant, and search for available flats based on your budget.
    
    To get started, you can explore the following pages:
    - **HDB Resale Chatbot**: Chat with our virtual assistant about the HDB resale process.
    - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.
    - **About Us**: Learn more about our project, objectives, and data sources.
    - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
    """)

elif page == "About Us":
    st.header("About Us")
    st.write("""
    This project aims to guide users through the process of buying an HDB flat in the resale market.
    Key features include an AI-powered chatbot to answer questions and a resale flat search based on budget.
    Data sources include official HDB websites and data.gov.sg.
    """)

elif page == "Methodology":
    st.header("Methodology")
    st.write("""
    This section describes the data flow and implementation of the app:
    
    1. **HDB Resale Chatbot**: 
        - User inputs a question → Query is passed to Langchain model → Retrieve relevant documents from the HDB website or fallback JSON → Generate response → Display to user.
    2. **HDB Resale Flat Search**: 
        - User selects budget → Data fetched from data.gov.sg → Filter flats within budget → Display relevant flats.
    """)
    
    # Flowchart for Use Cases (Illustrative Example)
    st.write("Flowchart for Use Cases:")
    st.image("methodology_flowchart.png")  # You can generate this from a tool like Graphviz.

elif page == "HDB Resale Chatbot":
    st.header("HDB Resale Chatbot")
    st.write("""
    Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
    Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, just ask me anything.
    I'm here to guide you every step of the way. Simply type your question below, and I'll provide you with the information you need.
    """)

    user_question = st.text_input("Ask a question about the HDB resale process:")
    if st.button("Submit"):
        if user_question:
            response = chain.invoke(user_question)
            st.write(response)
        else:
            st.write("Please enter a question to get started.")

elif page == "HDB Resale Flat Search":
    st.header("HDB Resale Flat Search")
    st.write("""
    This tool helps you find resale flats that fit your budget. Simply move the slider below to set your budget, and we will display relevant resale flats.
    """)
    
    # Budget slider
    budget = st.slider("Select your budget (SGD):", min_value=100000, max_value=2000000, step=50000, format="SGD {value:,.0f}")
    st.write(f"Your selected budget is: SGD {budget:,.0f}")

    # Fetch and display relevant flats (this is illustrative, you need to implement data fetching)
    st.write("Displaying resale flats within your budget...")
    # Example: Fetch flats based on budget and display as a table
    flats_data = pd.DataFrame({
        "Town": ["Town A", "Town B", "Town C"],
        "Flat Type": ["3-room", "4-room", "5-room"],
        "Resale Price (SGD)": ["$400,000", "$600,000", "$800,000"],
        "Storey Range": ["Low", "Mid", "High"],
        "Floor Area (sqm)": [60, 80, 100]
    })
    st.dataframe(flats_data)
