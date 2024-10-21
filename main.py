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
import altair as alt
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

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

# Function to generate historical resale price chart
def plot_resale_price_trend(df, town=None, flat_type=None):
    chart_data = df.copy()
    
    if town != "Any":
        chart_data = chart_data[chart_data['town'] == town]
    if flat_type != "Any":
        chart_data = chart_data[chart_data['flat_type'] == flat_type]
    
    line_chart = alt.Chart(chart_data).mark_line().encode(
        x='month:T',
        y='resale_price:Q',
        color='town:N',
        tooltip=['town', 'flat_type', 'resale_price', 'month']
    ).interactive()
    
    st.altair_chart(line_chart, use_container_width=True)

# Heatmap of resale prices
def create_heatmap(df):
    # Example lat/lng coordinates for towns (you'll need actual coordinates)
    town_coords = {
        'Ang Mo Kio': [1.3691, 103.8497],
        'Bedok': [1.3251, 103.9308],
        # Add other town coordinates
    }
    
    heatmap_data = []
    for town, coord in town_coords.items():
        avg_price = df[df['town'] == town]['resale_price'].mean()
        if not pd.isna(avg_price):
            heatmap_data.append(coord + [avg_price])
    
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11)
    HeatMap(heatmap_data, max_value=df['resale_price'].max()).add_to(m)
    
    folium_static(m)

# Comparison tool for resale flats
def compare_flats(filtered_df):
    selected_flats = st.multiselect("Select up to 3 flats for comparison", filtered_df['block'].unique(), max_selections=3)

    if selected_flats:
        comparison_df = filtered_df[filtered_df['block'].isin(selected_flats)]
        comparison_cols = ['Town', 'Flat Type', 'Resale Price', 'Floor Area Sqm', 'Storey Range', 'Remaining Lease']
        comparison_df['Resale Price'] = comparison_df['Resale Price'].apply(lambda x: f"${x:,.0f}")
        
        st.write("Comparing Selected Flats:")
        st.dataframe(comparison_df[comparison_cols].reset_index(drop=True))

# Handle content for each page
if not check_password():
    st.write("Please enter the correct password above to access the content.")
else:
    if page == "Home":
        st.write("""
            This application is designed to help you navigate the process of buying an HDB flat in the resale market.
            You can learn about the buying procedure, interact with a virtual assistant, and search for available flats based on your budget.
            
            To get started, you can explore the following pages:
            - **About Us**: Learn more about our project, objectives, and data sources.
            - **Methodology**: Understand the data flows and see the process flowcharts for each use case.
            - **HDB Resale Chatbot**: Chat with our virtual assistant about the HDB resale process.
            - **HDB Resale Flat Search**: Search for available resale flats based on your budget and preferences.
        """)
        
        with st.expander("Disclaimer"):
            st.write("""
            IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

            Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

            Always consult with qualified professionals for accurate and personalized advice.
            """)

    elif page == "About Us":
        st.write("""
            This project aims to guide users through the process of buying an HDB flat in the resale market.
            Key features include an AI-powered chatbot to answer questions and a resale flat search based on budget.
            Data sources include official HDB websites and data.gov.sg.
        """)

    elif page == "Methodology":
        st.write("""
            This section describes the data flow and implementation of the app:
            1. **HDB Resale Chatbot**: 
                - User inputs a question → Query is passed to Langchain model → Retrieve relevant documents from the HDB website or fallback JSON → Generate response → Display to user.
            2. **HDB Resale Flat Search**: 
                - User selects budget → Data fetched from CSV file → Filter flats within budget → Display relevant flats.
        """)
        # Flowchart for Use Cases (Illustrative Example)
        st.write("Flowchart for Use Cases:")
        st.image("methodology_flowchart.png")  # You can generate this from a tool like Graphviz.

    elif page == "HDB Resale Chatbot":
        st.write("""
            Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
            Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, I'm here to assist you!
        """)
        
        user_question = st.text_input("Type your question:")
        if st.button("Submit"):
            with st.spinner("Rina is thinking..."):
                response = chain.invoke(user_question)
                st.write(response)
                
            if "eligibility" in user_question.lower():
                st.image("eligibility_flowchart.png")

    elif page == "HDB Resale Flat Search":
        st.write("Search for resale flats based on your preferences and budget.")
        
        # Load your dataset here. For example:
        df = pd.read_csv("resale_flat_prices.csv")
        filtered_df = df  # Filter the dataframe based on user input

        # Historical resale price trends chart
        st.write("Historical Resale Price Trends")
        if st.button("Show Price Trends"):
            plot_resale_price_trend(df, town="Any", flat_type="Any")

        # Comparison tool
        compare_flats(filtered_df)

        # Heatmap
        if st.button("Show Heatmap of Resale Prices"):
            create_heatmap(df)
