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
import re

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = {'positive': 0, 'negative': 0}

# Function to scrape HDB website content
@st.cache_data(ttl=86400)
def scrape_hdb_resale_info():
    base_url = "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/buying-procedure-for-resale-flats"
    start_url = "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/buying-procedure-for-resale-flats/overview"
    output_file = 'hdb_resale_data.csv'
    visited = set()

    def scrape_page(url):
        if url in visited:
            return

        visited.add(url)

        try:
            response = requests.get(url, headers={'User-Agent': 'Custom Scraper 1.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Scrape content from the current page
            title = soup.find('h1').text.strip() if soup.find('h1') else 'No Title'
            content = soup.find('main').text.strip() if soup.find('main') else 'No Content'

            # Save scraped data to CSV
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([url, title, content])

            # Find all links on the page
            links = soup.find_all('a', href=True)

            for link in links:
                full_url = urljoin(base_url, link['href'])

                # Check if the link is within the same domain
                if full_url.startswith(base_url) and full_url not in visited:
                    # Delay to avoid overloading the server
                    time.sleep(1)
                    scrape_page(full_url)

        except requests.RequestException as e:
            st.error(f"Error scraping {url}: {e}")

    # Create or clear the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['URL', 'Title', 'Content'])

    scrape_page(start_url)

    # Read the CSV file and return the data
    scraped_data = []
    with open(output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            scraped_data.append({'url': row[0], 'title': row[1], 'content': row[2]})

    return scraped_data

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

# Function to handle user feedback
def handle_feedback(message_index, feedback):
    st.session_state.feedback[message_index] = feedback
    feedback_type = "positive" if feedback else "negative"
    st.session_state.feedback_count[feedback_type] += 1

    if feedback:
        st.success("Thank you for your positive feedback!")
    else:
        st.info("We're sorry to hear that. Could you tell us why the response was unsatisfactory?")
        
        # Structured feedback options
        feedback_reason = st.selectbox("Why was the response unsatisfactory?", 
                                       ["Too brief", "Too vague", "Not relevant", "Other"])
        if feedback_reason:
            detailed_feedback = st.text_area("Please provide additional details (optional):")
            store_detailed_feedback(message_index, feedback_reason, detailed_feedback)
        st.success("Thank you for your detailed feedback!")
    
    # Store feedback in CSV
    store_feedback(message_index, feedback, st.session_state.chat_history[message_index])

def store_detailed_feedback(message_index, feedback_reason, detailed_feedback):
    # Logic to store more detailed feedback
    with open("detailed_feedback.csv", "a", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["index", "feedback_reason", "detailed_feedback", "timestamp"])
        if os.stat("detailed_feedback.csv").st_size == 0:
            writer.writeheader()
        writer.writerow({
            "index": message_index,
            "feedback_reason": feedback_reason,
            "detailed_feedback": detailed_feedback,
            "timestamp": datetime.now().isoformat()
        })

def improve_chatbot_responses(message):
    # Analyze feedback from the CSV to detect common issues
    positive_count, negative_count = 0, 0
    with open("feedback_log.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["feedback"] == "True":
                positive_count += 1
            else:
                negative_count += 1

    total_feedback = positive_count + negative_count
    positive_ratio = positive_count / (total_feedback + 1e-6)  # Avoid division by zero

    if positive_ratio < 0.7:
        # Modify future responses to be more detailed or clear for similar queries
        st.warning("We've noticed a need for improvement in some responses. We'll adjust future replies to be more detailed.")

        # For simplicity, let's assume we're updating a "common issue" prompt
        common_issues = analyze_common_issues(message)
        new_template = f"You are Rina, an AI assistant guiding users through the HDB resale process. Be especially clear on {common_issues}."
        
        # Update the prompt chain dynamically
        prompt_template.template = new_template

def analyze_common_issues(message):
    # Dummy function to simulate analysis of common issues in messages
    # This can be extended to use NLP techniques to identify patterns in feedback
    if "eligibility" in message:
        return "eligibility criteria"
    elif "application" in message:
        return "application procedures"
    else:
        return "the resale process"

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
        st.write("""
        # Welcome to the HDB Resale Guide

        This application is designed to assist you in navigating the process of buying an HDB flat in the resale market. Whether you're a first-time buyer or looking to upgrade, our tools and resources are here to help you make informed decisions.

        ## Key Features:

        1. **HDB Resale Chatbot**: Interact with Rina, our AI assistant, to get answers to your questions about the HDB resale process.
        2. **HDB Resale Flat Search**: Find available resale flats based on your budget and preferences.
        3. **Comprehensive Information**: Access detailed guides and up-to-date information about the HDB resale market.

        ## How to Use This App:

        1. Explore the different pages using the sidebar navigation.
        2. Chat with Rina to get personalized answers to your HDB resale questions.
        3. Use the Resale Flat Search tool to find properties within your budget.
        4. Read through our methodology to understand how we process and present information.

        Get started by selecting a page from the sidebar or by asking Rina a question about HDB resale flats!
        """)

        st.info("""
        📌 Note: This application uses data from official HDB sources and is regularly updated to provide you with the most current information available.
        """)

        with st.expander("Disclaimer"):
            st.write("""
            IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

            Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

            Always consult with qualified professionals for accurate and personalized advice.
            """)

    elif page == "About Us":
        st.write("""
        # About Us

        ## Project Scope
        Our HDB Resale Guide is a comprehensive tool designed to simplify the process of buying a resale HDB flat in Singapore. We aim to provide accurate, up-to-date information and interactive tools to assist potential buyers in making informed decisions.

        ## Objectives
        1. To demystify the HDB resale process for potential buyers.
        2. To provide an AI-powered assistant capable of answering specific questions about HDB resale procedures.
        3. To offer a user-friendly interface for searching available HDB resale flats based on budget and preferences.
        4. To present data-driven insights into the HDB resale market trends.

        ## Data Sources
        Our application relies on the following data sources to ensure accuracy and relevance:
        - Official HDB website (www.hdb.gov.sg)
        - Data.gov.sg for HDB resale transaction data
        - HDB press releases and official announcements

        ## Features
        1. **HDB Resale Chatbot (Rina)**:
          - AI-powered assistant to answer user queries
          - Utilizes natural language processing to understand and respond to questions
          - Provides information based on the latest HDB guidelines and procedures

        2. **HDB Resale Flat Search**:
          - Interactive tool to search for resale flats within a specified budget
          - Filters for town, flat type, and other preferences
          - Displays relevant property details and transaction history

        3. **Data Visualizations**:
          - Price distribution charts
          - Average price by town comparisons
          - Price vs. floor area scatter plots

        4. **Methodology Transparency**:
          - Detailed explanation of our data processing and analysis methods
          - Flowcharts illustrating the application's processes

        We are committed to providing a valuable resource for anyone navigating the HDB resale market, combining the latest technology with comprehensive, reliable information.
        """)

    elif page == "Methodology":
        st.write("""
        # Methodology

        This section describes the data flow and implementation details of our HDB Resale Guide application. We use a combination of web scraping, data processing, and machine learning techniques to provide accurate and up-to-date information.

        ## 1. HDB Resale Chatbot

        Our AI assistant, Rina, uses a retrieval-augmented generation (RAG) approach to answer user queries:

        1. User inputs a question about the HDB resale process.
        2. The query is passed to our Langchain model.
        3. Relevant documents are retrieved from:
          - Scraped HDB website content
          - Fallback JSON document (if web scraping yields no results)
        4. The retrieved context and user query are used to generate a response.
        5. The response is displayed to the user in the chat interface.

        ### Data Flow:
        User Input → Query Processing → Document Retrieval → Context + Query to LLM → Response Generation → Display to User

        ## 2. HDB Resale Flat Search

        Our flat search feature processes data as follows:

        1. User selects budget and preferences (town, flat type).
        2. Data is fetched from a CSV file containing recent HDB resale transactions.
        3. The application filters flats within the specified budget and preferences.
        4. Matching flats are displayed in a table format.
        5. Additional visualizations (price distribution, average price by town, etc.) are generated using Plotly.

        ### Data Flow:
        User Input (Budget/Preferences) → Data Filtering → Sorting → Result Display → Data Visualization

        ## Implementation Details

        - Web Scraping: We use BeautifulSoup to scrape the official HDB website daily.
        - Data Processing: Pandas is used for data cleaning and manipulation.
        - Language Model: We utilize the GPT-4o-mini model via the Langchain library.
        - Visualization: Plotly Express is used for creating interactive charts and graphs.
        - Data Storage: Scraped data is stored in CSV format for easy access and processing.

        ## Security Measures

        - User inputs are sanitized to prevent injection attacks.
        - The application is password-protected to control access.
        - We do not store personal user data or chat histories beyond the current session.

        ## Limitations and Future Improvements

        - The chatbot's knowledge is limited to its training data and may not cover very recent changes.
        - The flat search feature relies on historical data and may not reflect real-time availability.
        - Future versions aim to incorporate real-time data feeds and more advanced predictive analytics.
        """)

        # Flowchart for Use Cases
        st.write("## Flowchart for Use Cases:")
        st.image("methodology_flowchart.png", caption="Detailed Flowchart of Application Processes")

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

        # Display chat history with improved formatting and feedback mechanism
        st.write("Chat History:")
        for i in range(len(st.session_state.chat_history) - 1, -1, -2):
            if i > 0:
                st.write("---")
                human_message = st.session_state.chat_history[i-1][1]
                rina_message = st.session_state.chat_history[i][1]
                st.write(f"**You:** {human_message}")
                st.write(f"**Rina:** {rina_message}")

                # Add thumbs up/down buttons for feedback
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        handle_feedback(i, True)
                with col2:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        handle_feedback(i, False)

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

                # Create a copy of the dataframe for display purposes
                display_df = filtered_df.copy()
                display_df['formatted_price'] = display_df['resale_price'].apply(lambda x: f"${x:,.0f}")
                display_df['month'] = display_df['month'].dt.strftime('%Y-%m')

                # Rename columns before selecting them
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
                fig_price.update_layout(
                    hovermode="x",
                    hoverlabel=dict(bgcolor="white", font_size=12),
                    xaxis=dict(rangeslider=dict(visible=True), type="linear")
                )
                fig_price.update_traces(hovertemplate="Price: $%{x:,.0f}<br>Count: %{y}")
                st.plotly_chart(fig_price)

                # Average Price by Town
                avg_price_by_town = filtered_df.groupby("town")["resale_price"].mean().sort_values(ascending=False)
                fig_town = px.bar(avg_price_by_town, x=avg_price_by_town.index, y="resale_price", title="Average Price by Town")
                fig_town.update_yaxes(title_text="Average Resale Price")
                fig_town.update_layout(
                    hovermode="x",
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )
                fig_town.update_traces(hovertemplate="Town: %{x}<br>Average Price: $%{y:,.0f}")
                st.plotly_chart(fig_town)

                # Price vs Floor Area
                fig_area = px.scatter(filtered_df, x="floor_area_sqm", y="resale_price", color="flat_type", title="Price vs Floor Area")
                fig_area.update_xaxes(title_text="Floor Area (sqm)")
                fig_area.update_yaxes(title_text="Resale Price")
                fig_area.update_layout(
                    hovermode="closest",
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )
                fig_area.update_traces(hovertemplate="Floor Area: %{x} sqm<br>Price: $%{y:,.0f}<br>Flat Type: %{customdata[0]}")
                fig_area.update_traces(customdata=filtered_df[['flat_type']])
                st.plotly_chart(fig_area)

            else:
                st.write("No flats found within your budget and preferences.")
