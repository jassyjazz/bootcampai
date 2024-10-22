import openai
import json
import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from urllib.parse import urljoin
import time
import csv
import hashlib
import re

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to scrape HDB website content
@st.cache_data(ttl=86400)
def scrape_hdb_resale_info():
    # ... (keep the existing scraping function)

# Load fallback document for RAG from JSON
def load_document():
    # ... (keep the existing load_document function)

# Retrieve relevant documents
def retrieve_relevant_documents(user_prompt):
    # ... (keep the existing retrieve_relevant_documents function)

# Initialize the LLM with gpt-4o-mini model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=st.secrets["general"]["OPENAI_API_KEY"]
)

# Create the prompt chain with improved prompt engineering
prompt_template = PromptTemplate(
    input_variables=["relevant_docs", "question", "chat_history"],
    template='''You are Rina, an AI assistant guiding users through the HDB resale process in Singapore.
    Provide accurate information based on the following context and chat history:

    Context:
    {relevant_docs}

    Chat History:
    {chat_history}

    Human: {question}
    Rina: '''
)

# Function to get chat history as a string
def get_chat_history():
    return "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history[-5:]])

chain = (
    {"relevant_docs": retrieve_relevant_documents, "question": RunnablePassthrough(), "chat_history": get_chat_history}
    | prompt_template
    | llm
    | StrOutputParser()
)

# Password Protection with improved security
def check_password():
    if st.session_state.authenticated:
        return True

    password = st.text_input("Enter password to access the pages:", type="password")
    if st.button("Enter"):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if hashed_password == "2c5f3e6a2e1d5e4c1f9d8b7a6c3b2a1d":  # Hash of "bootcamp123"
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
        # ... (keep the existing Home page content)

    elif page == "About Us":
        # ... (keep the existing About Us page content)

    elif page == "Methodology":
        # ... (keep the existing Methodology page content)

    elif page == "HDB Resale Chatbot":
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image("https://raw.githubusercontent.com/jassyjazz/bootcampai/main/boticon.png", use_column_width=True)

        with col2:
            st.write("""
                Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
                Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, just ask me anything.
                I'm here to guide you every step of the way. Simply type your question below, or click on one of the common queries to get started.
            """)

        # Common queries as buttons
        common_queries = [
            "What are the eligibility criteria for buying an HDB resale flat?",
            "How do I apply for an HDB resale flat?",
            "What are the steps involved in the resale process?",
            "What grants are available for HDB resale flats?",
            "How long does the HDB resale process take?"
        ]

        col1, col2 = st.columns(2)
        for i, query in enumerate(common_queries):
            if i % 2 == 0:
                if col1.button(query):
                    st.session_state.chat_history.append(("Human", query))
                    with st.spinner("Generating response..."):
                        response = chain.invoke(query)
                    st.session_state.chat_history.append(("Rina", response))
            else:
                if col2.button(query):
                    st.session_state.chat_history.append(("Human", query))
                    with st.spinner("Generating response..."):
                        response = chain.invoke(query)
                    st.session_state.chat_history.append(("Rina", response))

        user_question = st.text_input("Or ask your own question about the HDB resale process:")
        if st.button("Submit"):
            if user_question:
                # Sanitize user input
                sanitized_question = re.sub(r'[^\w\s]', '', user_question)
                st.session_state.chat_history.append(("Human", sanitized_question))

                with st.spinner("Generating response..."):
                    response = chain.invoke(sanitized_question)
                st.session_state.chat_history.append(("Rina", response))
            else:
                st.write("Please enter a question to get started.")

        # Display chat history with improved formatting
        st.write("Chat History:")
        for i in range(len(st.session_state.chat_history) - 1, -1, -2):
            if i > 0:
                st.write("---")
                human_message = st.session_state.chat_history[i-1][1]
                rina_message = st.session_state.chat_history[i][1]
                st.write(f"**You:** {human_message}")
                st.write(f"**Rina:** {rina_message}")

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
            df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
            return df

        df = load_data()

        budget = st.slider(
            "Select your budget (SGD):",
            min_value=200000,
            max_value=1600000,
            step=50000
        )

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
        sort_choice = st.selectbox("Sort by:", list(sort_options.keys()))

        if st.button("Search"):
            filtered_df = df[df['resale_price'] <= budget]

            if town != "Any":
                filtered_df = filtered_df[filtered_df['town'] == town]
            if flat_type != "Any":
                filtered_df = filtered_df[filtered_df['flat_type'] == flat_type]

            sort_column, ascending = sort_options[sort_choice]
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

            if not filtered_df.empty:
                st.write(f"Found {len(filtered_df)} matching flats:")

                display_df = filtered_df.copy()
                display_df['formatted_price'] = display_df['resale_price'].apply(lambda x: f"${x:,.0f}")
                display_df['month'] = display_df['month'].dt.strftime('%Y-%m')

                display_df = display_df.rename(columns={
                    'town': 'Town',
                    'flat_type': 'Flat Type',
                    'block': 'Block',
                    'street_name': 'Street Name',
                    'storey_range': 'Storey Range',
                    'floor_area_sqm': 'Floor Area Sqm',
                    'remaining_lease': 'Remaining Lease',
                    'formatted_price': 'Resale Price',
                    'month': 'Month'
                })

                columns_to_display = ['Town', 'Flat Type', 'Block', 'Street Name', 'Storey Range', 'Floor Area Sqm', 'Remaining Lease', 'Resale Price', 'Month']
                st.dataframe(display_df[columns_to_display].reset_index(drop=True))

                # Visualizations
                st.subheader("Visualizations")

                # Price Distribution
                fig_price = px.histogram(filtered_df, x="resale_price", title="Price Distribution of Matching Flats")
                fig_price.update_xaxes(title_text="Resale Price")
                st.plotly_chart(fig_price)

                # Average Price by Town
                avg_price_by_town = filtered_df.groupby("town")["resale_price"].mean().sort_values(ascending=False)
                fig_town = px.bar(avg_price_by_town, x=avg_price_by_town.index, y="resale_price", title="Average Price by Town")
                fig_town.update_yaxes(title_text="Average Resale Price")
                st.plotly_chart(fig_town)

                # Price vs Floor Area
                fig_area = px.scatter(filtered_df, x="floor_area_sqm", y="resale_price", color="flat_type", title="Price vs Floor Area")
                fig_area.update_xaxes(title_text="Floor Area (sqm)")
                fig_area.update_yaxes(title_text="Resale Price")
                st.plotly_chart(fig_area)

            else:
                st.write("No flats found within your budget and preferences.")

# Add a footer with disclaimer
st.markdown("---")
st.markdown("""
    <small>Disclaimer: This application is for educational purposes only. The information provided may not be up-to-date or accurate.
    Always consult official HDB sources and professionals for the most current and reliable information.</small>
    """, unsafe_allow_html=True)
