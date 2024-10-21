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
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough()}
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

# Function to plot resale prices by town and flat type
def plot_resale_prices(filtered_df):
    fig = px.bar(filtered_df,
                 x='town', 
                 y='resale_price',
                 color='flat_type', 
                 title="Resale Prices by Town and Flat Type",
                 labels={'resale_price': 'Resale Price (SGD)', 'town': 'Town'},
                 color_discrete_sequence=px.colors.qualitative.Set1)  # Clear color distinction
                 
    # Adjust layout for better readability
    fig.update_layout(
        xaxis_title="Town",
        yaxis_title="Resale Price (SGD)",
        title_font=dict(size=20),
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",  # Group hover information
        showlegend=True
    )
    
    # Add tooltips with additional information
    fig.update_traces(
        hovertemplate="<b>Town:</b> %{x}<br><b>Flat Type:</b> %{legendgroup}<br><b>Price:</b> %{y:,.0f} SGD<br><b>Floor Area:</b> %{customdata[0]} sqm"
    )

    return fig
    
# Handle content for each page
if not check_password():
    st.write("Please enter the correct password above to access the content.")
else:
    if page == "Home":
        st.write("""This application is designed to help you navigate the process of buying an HDB flat in the resale market.
            You can learn about the buying procedure, interact with a virtual assistant, and search for available flats based on your budget.
            To get started, you can explore the following pages:
            - **About Us**: Learn more about our project, objectives, and data sources.
            - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
            - **HDB Resale Chatbot**: Chat with our virtual assistant about the HDB resale process.
            - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.
        """)
        with st.expander("Disclaimer"):
            st.write("""IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
            Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.
            Always consult with qualified professionals for accurate and personalized advice.
            """)
        
    elif page == "HDB Resale Chatbot":
        st.write("""Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
            Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, just ask me anything.
            I'm here to guide you every step of the way. Simply type your question below, and I'll provide you with the information you need.
        """)
        
        common_queries = [
            "What are the eligibility criteria for buying an HDB resale flat?",
            "How do I apply for a resale flat?",
            "What are the steps involved in the resale process?",
            "How is the resale price determined?"
        ]
        
        # Display buttons for common queries
        for query in common_queries:
            if st.button(query):
                user_question = query
                with st.spinner("Generating response..."):
                    response = chain.invoke(user_question)
                st.write(response)
        
        user_question = st.text_input("Ask a question about the HDB resale process:")
        if st.button("Submit") and user_question:
            sanitized_question = user_question.replace("{", "").replace("}", "").replace("\\", "")
            with st.spinner("Generating response..."):
                response = chain.invoke(sanitized_question)
            st.write(response)
        
    elif page == "HDB Resale Flat Search":
        st.write("""
            This tool allows you to search for available HDB resale flats within your budget. Simply adjust the budget slider, select your preferred town and flat type, 
            and the app will display matching resale flats based on data from a recent HDB resale transactions dataset.
        """)
    
        @st.cache_data
        def load_data():
            url = "https://raw.githubusercontent.com/jassyjazz/bootcampai/main/resaleflatprices.csv"
            df = pd.read_csv(url)
            df['month'] = pd.to_datetime(df['month'])
            return df
    
        df = load_data()
    
        budget = st.slider(
            "Select your budget (SGD):",
            min_value=200000,
            max_value=1000000,
            value=(300000, 600000),
            step=10000,
            format="SGD {value:,}",
        )
    
        # Filter based on budget
        filtered_df = df[df['resale_price'].between(budget[0], budget[1])]
    
        plot_resale_prices(filtered_df)
        st.write(filtered_df[['town', 'flat_type', 'resale_price', 'storey_range', 'floor_area_sqm']].style.format({
            'resale_price': '${:,.0f}'
        }))
    
    elif page == "About Us":
        st.write("""This project aims to make the process of buying an HDB resale flat more understandable and accessible. We aggregate official data and provide an 
        interactive and informative platform for users interested in purchasing an HDB flat. Our data sources include **data.gov.sg**, **HDB**, and other open data platforms.""")

    elif page == "Methodology":
        st.write("""In this section, we explain how the data flows through the system and how we implement each use case. We present process flowcharts and details 
        about the system's components, including the **data fetching**, **scraping**, **processing**, and **response generation** aspects of the project.""")
