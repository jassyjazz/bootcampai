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

# Password Protection
def check_password():
    password = st.text_input("Enter password to access the main page:", type="password")
    if password != "bootcamp123" and password != "":
        st.error("Incorrect password. Please try again.")
        return False
    return True

# Load page content after password is validated
if not check_password():
    st.stop()  # Stop further execution if password is incorrect

# Page selection after password validation
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
    st.write("""This application is designed to help you navigate the process of buying an HDB flat in the resale market.
                You can learn about the buying procedure, interact with a virtual assistant, and search for available flats based on your budget.
                To get started, you can explore the following pages:
                - **HDB Resale Chatbot**: Chat with our virtual assistant about the HDB resale process.
                - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.
                - **About Us**: Learn more about our project, objectives, and data sources.
                - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
                """)
elif page == "About Us":
    st.header("About Us")
    st.write("""This project aims to guide users through the process of buying an HDB flat in the resale market.
                Key features include an AI-powered chatbot to answer questions and a resale flat search based on budget.
                Data sources include official HDB websites and data.gov.sg.
                """)
elif page == "Methodology":
    st.header("Methodology")
    st.write("""This section describes the data flow and implementation of the app:
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
    st.write("""Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
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
    st.write("""This tool allows you to search for available HDB resale flats within your budget. Simply adjust the budget slider, select your preferred town and flat type, 
    and the app will display matching resale flats based on data from data.gov.sg.
    """)

    # Personalizing flat search with user inputs
    budget = st.slider(
    "Select your budget (SGD):",
    min_value=100000,
    max_value=2000000,
    step=50000
    )

    # Display the formatted budget with commas as thousand separators and currency symbol
    formatted_budget = f"SGD ${budget:,.0f}"
    st.write(f"Your selected budget: {formatted_budget}")
    town = st.selectbox("Select your preferred town:", ["Any", "Ang Mo Kio", "Bedok", "Bukit Merah", "Bukit Panjang", "Choa Chu Kang", "Hougang", "Jurong East"])
    flat_type = st.selectbox("Select flat type:", ["Any", "2 Room", "3 Room", "4 Room", "5 Room", "Executive"])

    def get_resale_flats_by_budget(budget, town, flat_type):
        datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
        url = f"https://data.gov.sg/api/action/datastore_search?resource_id={datasetId}&limit=100"
        
        try:
            response = requests.get(url)
            data = response.json()
            flats = data["result"]["records"]
            filtered_flats = []

            for flat in flats:
                price = int(flat["resale_price"])
                if price <= budget and (town == "Any" or flat["town"] == town) and (flat_type == "Any" or flat["flat_type"] == flat_type):
                    filtered_flats.append(flat)
            
            return filtered_flats
        except Exception as e:
            st.error(f"Error fetching resale flat data: {str(e)}")
            return []

    filtered_flats = get_resale_flats_by_budget(budget, town, flat_type)
    if filtered_flats:
        st.write(f"Found {len(filtered_flats)} matching flats:")
        flats_df = pd.DataFrame(filtered_flats)
        st.dataframe(flats_df)
    else:
        st.write("No flats found within your budget and preferences.")
