import openai
import json
import streamlit as st
import os
import pandas as pd
import requests

# Set OpenAI API key from Streamlit secrets
api_key = st.secrets["general"]["OPENAI_API_KEY"]
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

# Function for password protection
def password_protect():
    password = st.text_input("Enter password to access the app:", type="password")
    if password:
        if password != "bootcamp123":
            st.error("Incorrect password. Please try again.")
            return False
        else:
            return True
    return False

# Home Page
if not password_protect():
    st.stop()  # Stop the execution if password is incorrect or not entered

# Remove the redundant title by setting it only once per page
st.title("HDB Resale Guide")

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

# Streamlit interface

# Main page
page = st.sidebar.selectbox("Select a page:", ["Home", "HDB Resale Chatbot", "HDB Resale Flat Search", "About Us", "Methodology"])

if page == "Home":
    st.header("Welcome to the HDB Resale Guide")
    st.write("""
    Welcome! This app is designed to assist you in navigating the process of buying a resale HDB flat. Whether you're new to the process or seeking guidance, 
    this app provides you with the tools you need. You can ask questions through the chatbot, explore resale flats based on your budget, and understand the 
    entire HDB resale process step by step.
    """)

elif page == "HDB Resale Chatbot":
    st.header("Hi! I am your virtual HDB assistant")
    st.write("""
    I'm here to help you with any questions you have about the HDB resale process. Whether you're wondering about eligibility, 
    how to apply for a resale flat, or the procedures involved, just ask me anything. I'm here to guide you every step of the way.
    """)

    user_question = st.text_input("Ask a question about the HDB resale process:")
    
    if user_question:
        response = retrieve_relevant_documents(user_question)
        st.write(response)

elif page == "HDB Resale Flat Search":
    st.header("Find HDB Resale Flats Within Your Budget")
    st.write("""
    This page allows you to search for resale flats based on your preferred budget and flat criteria. Simply adjust the budget slider, 
    select your preferred town and flat type, and the app will display matching resale flats based on the data available from data.gov.sg.
    """)

    # Personalizing flat search with user inputs
    budget = st.slider("Select your budget (SGD):", min_value=100000, max_value=2000000, step=50000)
    # Format the displayed value as $x,xxx
    formatted_budget = f"${budget:,}"
    st.write(f"Your selected budget: {formatted_budget}")

    town = st.selectbox("Select your preferred town:", ["Any", "Ang Mo Kio", "Bedok", "Bukit Merah", "Bukit Panjang", "Choa Chu Kang", "Hougang", "Jurong East"])
    flat_type = st.selectbox("Select flat type:", ["Any", "2 Room", "3 Room", "4 Room", "5 Room", "Executive"])
    
    datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={datasetId}"

    def get_resale_flats_by_budget(budget, town, flat_type):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()['result']['records']
            filtered_flats = []
            for flat in data:
                # Filter based on user budget, town, and flat type
                if int(flat['resale_price']) <= budget and (town == "Any" or flat['town'] == town) and (flat_type == "Any" or flat['flat_type'] == flat_type):
                    filtered_flats.append(flat)
            
            # Display data in a table with formatted currency
            if filtered_flats:
                df = pd.DataFrame(filtered_flats, columns=["town", "flat_type", "resale_price", "storey_range", "floor_area_sqm"])
                df["resale_price"] = df["resale_price"].apply(lambda x: f"${int(x):,}")
                st.dataframe(df, width=1000)  # Set a larger table width
            else:
                st.write("No flats found matching your criteria.")  # Fixed the string error
        except Exception as e:
            st.error(f"Error fetching flats data: {str(e)}")
    
    # Execute flat search based on user inputs
    get_resale_flats_by_budget(budget, town, flat_type)

elif page == "About Us":
    st.header("About Us")
    st.write("""
    This application is designed to help users understand the process of buying a resale HDB flat in Singapore. With interactive tools like the resale chatbot 
    and flat search, we aim to provide a comprehensive guide to navigating the complexities of the HDB resale market.
    """)

elif page == "Methodology":
    st.header("Methodology")
    st.write("""
    This section explains the data flows and implementation details behind the app. Here, we outline the process for each use case and include flowcharts for clarity.
    """)

# End of the application
