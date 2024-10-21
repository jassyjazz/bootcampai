import openai
import json
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import plotly.express as px

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

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

# Load resale flat data from your CSV file
@st.cache_data
def load_resale_data():
    url = "https://raw.githubusercontent.com/jassyjazz/bootcampai/main/resaleflatprices.csv"
    df = pd.read_csv(url)
    df['month'] = pd.to_datetime(df['month'])
    return df

# Fetch coordinates (dummy example, replace with actual API or dataset if possible)
@st.cache_data
def get_coordinates():
    coordinates_data = {
        "Ang Mo Kio": (1.3691, 103.8454),
        "Bedok": (1.3236, 103.9273),
        "Bishan": (1.3516, 103.8485),
        # Add more towns as needed
    }
    return coordinates_data

# Interactive heatmap of HDB resale transactions
def show_heatmap(df):
    coordinates_data = get_coordinates()

    df['latitude'] = df['town'].map(lambda x: coordinates_data.get(x, (None, None))[0])
    df['longitude'] = df['town'].map(lambda x: coordinates_data.get(x, (None, None))[1])
    
    df_filtered = df.dropna(subset=['latitude', 'longitude'])
    
    fig = px.density_mapbox(df_filtered, lat='latitude', lon='longitude', z='resale_price',
                            hover_name='town', hover_data=['flat_type', 'floor_area_sqm'],
                            radius=10, mapbox_style="stamen-terrain",
                            title="Heatmap of HDB Resale Transactions")
    fig.update_layout(mapbox=dict(center=dict(lat=1.3521, lon=103.8198), zoom=10))
    st.plotly_chart(fig)

# Password protection logic and page content
if not check_password():
    st.write("Please enter the correct password above to access the content.")
else:
    # Show the page selection sidebar
    page = st.sidebar.selectbox("Select a page", ["Home", "About Us", "Methodology", "HDB Resale Chatbot", "HDB Resale Flat Search"])

    # Dynamic title based on the selected page
    st.title(f"{page} - HDB Resale Guide")

    # Handle content for each page
    if page == "Home":
        st.write("""
            This application is designed to help you navigate the process of buying an HDB flat in the resale market.
            Explore the pages to learn about the process, interact with a virtual assistant, and search for flats based on your budget.
        """)
    elif page == "HDB Resale Flat Search":
        df = load_resale_data()

        st.write("Search for available HDB resale flats within your budget.")
        budget = st.slider("Select your budget (SGD):", min_value=200000, max_value=1600000, step=50000)
        town = st.selectbox("Select your preferred town:", ["Any"] + sorted(df['town'].unique().tolist()))
        flat_type = st.selectbox("Select flat type:", ["Any"] + sorted(df['flat_type'].unique().tolist()))
        
        filtered_df = df[df['resale_price'] <= budget]
        if town != "Any":
            filtered_df = filtered_df[filtered_df['town'] == town]
        if flat_type != "Any":
            filtered_df = filtered_df[filtered_df['flat_type'] == flat_type]
        
        if not filtered_df.empty:
            st.write(f"Found {len(filtered_df)} matching flats:")
            st.dataframe(filtered_df[['town', 'flat_type', 'resale_price', 'floor_area_sqm', 'month']])
            
            # Show heatmap of transactions
            show_heatmap(filtered_df)
        else:
            st.write("No flats found within your budget and preferences.")
