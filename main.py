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
from datetime import datetime
import plotly.express as px

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

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
    template='You are Rina, an AI assistant guiding users through the HDB resale process in Singapore. '
             'Provide accurate information based on the following context:\n\n'
             '{relevant_docs}\n\n'
             'Human: {question}\n'
             'Rina: '
)

chain = (
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough() }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Password Protection
def check_password():
    if st.session_state.authenticated:
        return True
    
    password = st.text_input("Enter password to access the pages:", type="password")
    if st.button("Enter"):
        if password == "bootcamp123":
            st.session_state.authenticated = True
            return True
        else:
            st.error("Please enter the correct password to access the content.")
    return False

# Show the page selection sidebar
page = st.sidebar.selectbox("Select a page", ["Home", "About Us", "Methodology", "HDB Resale Chatbot", "HDB Resale Flat Search"])

# Dynamic title based on the selected page
st.title(f"{page} - HDB Resale Guide")

# Handle content for each page
if not check_password():
    st.write("Please enter the correct password above to access the content.")
else:
    if page == "Home":
        st.write("""This application is designed to help you navigate the process of buying an HDB flat in the resale market. You can learn about the buying procedure, interact with a virtual assistant, and search for available flats based on your budget. 
        To get started, you can explore the following pages:
        - **About Us**: Learn more about our project, objectives, and data sources.
        - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
        - **HDB Resale Chatbot**: Chat with our virtual assistant about the HDB resale process.
        - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.""")

    elif page == "About Us":
        st.write("""### About Us
        Welcome to the HDB Resale Guide project! This app is designed to provide a step-by-step guide to buying an HDB flat in Singapore, specifically in the resale market. Our goal is to provide users with a clear, easy-to-understand interface to navigate the complex process of purchasing a resale flat.
        
        **Objectives:**
        - Simplify the HDB resale process.
        - Provide a personalized user experience.
        - Integrate official data sources to ensure accuracy and relevancy.
        
        **Data Sources:**
        - Data from the official HDB website.
        - Resale flat price data from the data.gov.sg API.""")

    elif page == "Methodology":
        st.write("""### Methodology
        The methodology behind this project includes the integration of various official data sources, scraping of relevant information, and the use of AI-driven chat functionalities to assist users. Here are the key components:
        
        **1. Data Collection:**
        - Resale flat prices are obtained via the data.gov.sg API.
        - The HDB website is scraped for detailed instructions on the resale process.
        
        **2. AI Chatbot:**
        - The AI assistant, powered by the GPT-4 model, answers users' questions about the HDB resale process based on retrieved documents.
        
        **3. User Experience:**
        - The app includes an interactive flat search feature, allowing users to filter resale flats based on their preferences and budget.
        - Dynamic charts display trends in resale prices to give users insights into the current market.
        
        **4. Process Flowcharts:**
        - Flowcharts are included to visually depict each step of the resale flat buying process.
        - A detailed breakdown of the methods used to fetch, filter, and process the data will be available soon.""")

    elif page == "HDB Resale Chatbot":
        st.write("""Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process. Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, just ask me anything. I'm here to guide you every step of the way. Simply type your question below, and I'll provide you with the information you need.""")
        
        user_question = st.text_input("Ask a question about the HDB resale process:")
        if st.button("Submit"):
            if user_question:
                sanitized_question = user_question.replace("{", "").replace("}", "").replace("\\", "")
                with st.spinner("Generating response..."):
                    response = chain.invoke(sanitized_question)
                st.write(response)

    elif page == "HDB Resale Flat Search":
        st.write("""This tool allows you to search for available HDB resale flats within your budget. Simply adjust the budget slider, select your preferred town and flat type, and the app will display matching resale flats based on data from a recent HDB resale transactions dataset.""")
        
        @st.cache_data
        def load_data():
            url = "https://raw.githubusercontent.com/jassyjazz/bootcampai/main/resaleflatprices.csv"
            df = pd.read_csv(url)
            df['month'] = pd.to_datetime(df['month'])
            return df

        df = load_data()

        budget = st.slider("Select your budget (SGD):", min_value=200000, max_value=1600000, step=50000)
        formatted_budget = f"SGD ${budget:,.0f}"
        st.write(f"Your selected budget: {formatted_budget}")
        
        town = st.selectbox("Select your preferred town:", ["Any"] + sorted(df['town'].unique().tolist()))
        flat_type = st.selectbox("Select flat type:", ["Any"] + sorted(df['flat_type'].unique().tolist()))
        
        sort_options = {
            "Newest to Oldest": ("month", False),
            "Oldest to Newest": ("month", True),
            "Price: Lowest to Highest": ("resale_price", True),
            "Price: Highest to Lowest": ("resale_price", False)
        }
        sort_choice = st.selectbox("Sort flats by:", list(sort_options.keys()))
        sort_column, sort_ascending = sort_options[sort_choice]
        
        filtered_data = df[df['resale_price'] <= budget]
        if town != "Any":
            filtered_data = filtered_data[filtered_data['town'] == town]
        if flat_type != "Any":
            filtered_data = filtered_data[filtered_data['flat_type'] == flat_type]
        
        sorted_data = filtered_data.sort_values(by=sort_column, ascending=sort_ascending)
        
        st.write("Resale Flats Available:")
        st.write(f"Displaying {len(sorted_data)} flats within your budget.")
        
        st.write(sorted_data[['town', 'flat_type', 'resale_price', 'storey_range', 'floor_area_sqm']].head(10))

        st.write(f"Here is a chart of resale prices in your selected town and flat type:")
        fig = px.box(sorted_data, x="flat_type", y="resale_price", title="Resale Price Distribution")
        st.plotly_chart(fig)

# Finalizing the requirements
