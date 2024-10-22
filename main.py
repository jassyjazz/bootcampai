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
from functools import lru_cache
from collections import Counter, defaultdict
import plotly.graph_objects as go

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = {'positive': 0, 'negative': 0}
if 'feedback_types' not in st.session_state:
    st.session_state.feedback_types = defaultdict(int)
if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = {}

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
@st.cache_data
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
@lru_cache(maxsize=100)
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
    temperature=0.7,
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

# Function to log feedback
def log_feedback(message, response, feedback, feedback_type=None, detailed_feedback=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_entry = {
        "timestamp": timestamp,
        "message": message,
        "response": response,
        "feedback": feedback,
    }
    if not feedback:  # Only include feedback_type for negative feedback
        feedback_entry["feedback_type"] = feedback_type
    if detailed_feedback:
        feedback_entry["detailed_feedback"] = detailed_feedback

    with open('feedback_log.json', 'a') as f:
        json.dump(feedback_entry, f)
        f.write('\n')

# Function to handle user feedback
def handle_feedback(message_index, feedback):
    if feedback:  # Positive feedback
        st.session_state.feedback[message_index] = True
        st.session_state.feedback_count['positive'] += 1
        log_feedback(st.session_state.chat_history[message_index][0],
                     st.session_state.chat_history[message_index][1],
                     True)
        st.success("Thank you for your positive feedback!")
    else:  # Negative feedback
        st.session_state.feedback[message_index] = False
        st.session_state.feedback_count['negative'] += 1

        # Store the message index for pending negative feedback
        st.session_state.pending_feedback[message_index] = {
            'submitted': False,
            'type': None,
            'details': None
        }

# New function to handle negative feedback submission
def submit_negative_feedback(message_index):
    feedback_type = st.session_state[f"feedback_type_{message_index}"]
    detailed_feedback = st.session_state[f"detailed_feedback_{message_index}"]

    st.session_state.feedback_types[feedback_type] += 1
    st.session_state.pending_feedback[message_index]['submitted'] = True
    st.session_state.pending_feedback[message_index]['type'] = feedback_type
    st.session_state.pending_feedback[message_index]['details'] = detailed_feedback

    log_feedback(st.session_state.chat_history[message_index][0],
                 st.session_state.chat_history[message_index][1],
                 False, feedback_type, detailed_feedback)
    st.success("Thank you for your detailed feedback. We'll use it to improve our responses.")

# Update the analyse_feedback function
def analyse_feedback():
    st.subheader("Feedback Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Feedback", st.session_state.feedback_count['positive'])
    with col2:
        st.metric("Negative Feedback", st.session_state.feedback_count['negative'])

    if st.session_state.feedback_count['negative'] > 0:
        st.warning(f"There are {st.session_state.feedback_count['negative']} instances of negative feedback. Here's the breakdown:")

        # Display feedback types
        for feedback_type, count in st.session_state.feedback_types.items():
            st.text(f"- {feedback_type}: {count}")

    st.info("We're using this feedback to improve our responses.")

    # Use feedback to improve chatbot responses
    improve_chatbot_responses()

# Function to improve chatbot responses based on feedback
def improve_chatbot_responses():
    total_feedback = sum(st.session_state.feedback_count.values())
    if total_feedback > 0:
        positive_ratio = st.session_state.feedback_count['positive'] / total_feedback

        global prompt_template
        if positive_ratio < 0.7:  # If less than 70% positive feedback
            st.warning("We've noticed that our responses could be improved. We're adjusting our system.")

            # Adjust the prompt template based on feedback types
            if st.session_state.feedback_types['Too brief'] > st.session_state.feedback_types['Too vague']:
                prompt_template = PromptTemplate(
                    input_variables=["relevant_docs", "question"],
                    template='You are Rina, an AI assistant guiding users through the HDB resale process in Singapore. '
                             'Provide detailed and comprehensive information based on the following context. '
                             'Ensure your responses are thorough and cover all relevant aspects:\n\n'
                             '{relevant_docs}\n\n'
                             'Human: {question}\n'
                             'Rina: '
                )
            elif st.session_state.feedback_types['Too vague'] > st.session_state.feedback_types['Too brief']:
                prompt_template = PromptTemplate(
                    input_variables=["relevant_docs", "question"],
                    template='You are Rina, an AI assistant guiding users through the HDB resale process in Singapore. '
                             'Provide clear, specific, and concrete information based on the following context. '
                             'Use examples and precise details where possible:\n\n'
                             '{relevant_docs}\n\n'
                             'Human: {question}\n'
                             'Rina: '
                )
            elif st.session_state.feedback_types['Not relevant'] > 0:
                prompt_template = PromptTemplate(
                    input_variables=["relevant_docs", "question"],
                    template='You are Rina, an AI assistant guiding users through the HDB resale process in Singapore. '
                             'Ensure your responses are highly relevant to the user\'s question. '
                             'Focus on addressing the specific query based on this context:\n\n'
                             '{relevant_docs}\n\n'
                             'Human: {question}\n'
                             'Rina: '
                )
        else:
            # Reset to default prompt if feedback is generally positive
            prompt_template = PromptTemplate(
                input_variables=["relevant_docs", "question"],
                template='You are Rina, an AI assistant guiding users through the HDB resale process in Singapore. '
                         'Provide accurate, concise, and helpful information based on the following context. '
                         'If you\'re unsure about any details, say so clearly:\n\n'
                         '{relevant_docs}\n\n'
                         'Human: {question}\n'
                         'Rina: '
            )

# Function to analyse feedback
def analyse_feedback():
    st.subheader("Feedback Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Feedback", st.session_state.feedback_count['positive'])
    with col2:
        st.metric("Negative Feedback", st.session_state.feedback_count['negative'])

    if st.session_state.feedback_count['negative'] > 0:
        st.warning(f"There are {st.session_state.feedback_count['negative']} instances of negative feedback. Here's the breakdown:")

        # Display feedback types
        for feedback_type, count in st.session_state.feedback_types.items():
            st.text(f"- {feedback_type}: {count}")

    st.info("We're using this feedback to improve our responses.")

    # Display recent feedback (both positive and negative)
    st.subheader("Recent Feedback")
    with open('feedback_log.json', 'r') as f:
        feedback_data = [json.loads(line) for line in f]
    for item in feedback_data[-5:]:  # Show last 5 feedback entries
        with st.expander(f"Feedback from {item['timestamp']}"):
            st.text(f"User: {item['message']}")
            st.text(f"Rina: {item['response']}")
            st.text(f"Feedback: {'Positive' if item['feedback'] else 'Negative'}")
            if not item['feedback']:  # Only show feedback type for negative feedback
                st.text(f"Feedback Type: {item.get('feedback_type', 'Not specified')}")
            if item.get('detailed_feedback'):
                st.text(f"Detailed Feedback: {item['detailed_feedback']}")

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
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://raw.githubusercontent.com/jassyjazz/bootcampai/main/hdb_logo.png", width=100)
    
        with col2:
            st.markdown(
                """
                <div style='display: flex; align-items: center;'>
                    <p style='margin: 0; font-size: 18px;'> <!-- Adjust font size as needed -->
                    This application is designed to assist you in navigating the process of buying an HDB flat in the resale market. Whether you're a first-time buyer or looking to upgrade, our tools and resources are here to help you make informed decisions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("## Key Features:")
        st.write("""
        1. **HDB Resale Chatbot**: Interact with Rina, our AI assistant, to get answers to your questions about the HDB resale process.
        2. **HDB Resale Flat Search**: Find available resale flats based on your budget and preferences.
        3. **Comprehensive Information**: Access detailed guides and up-to-date information about the HDB resale market.
            """)

        st.write("## How to Use This App:")
        st.write("""      
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
        # Custom CSS for styling the About Us page
        st.markdown(
            """
            <style>
            .about-title {
                color: #2B2D42;
                font-size: 36px;
                text-align: center;
                margin-bottom: 20px;
            }
            .about-section {
                background-color: #ffffff; 
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .about-list {
                margin-left: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Section 1: Project Scope
        st.markdown("<div class='about-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Project Scope</h2>", unsafe_allow_html=True)
        st.write("""
        HDB Resale Guide is a comprehensive tool designed to simplify the process of buying a resale HDB flat in Singapore. We aim to provide accurate, up-to-date information and interactive tools to assist potential buyers in making informed decisions.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 2: Objectives
        st.markdown("<div class='about-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Objectives</h2>", unsafe_allow_html=True)
        st.write("Our main objectives include:")
        objectives = [
            "To simplify the HDB resale process for potential buyers.",
            "To provide an AI-powered assistant capable of answering specific questions about HDB resale procedures.",
            "To offer a user-friendly interface for searching available HDB resale flats based on budget and preferences.",
            "To present data-driven insights into the HDB resale market trends."
        ]
        for obj in objectives:
            st.markdown(f"- {obj}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 3: Data Sources
        st.markdown("<div class='about-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Data Sources</h2>", unsafe_allow_html=True)
        st.write("""
        Our application relies on the following data sources to ensure accuracy and relevance:
        - Official HDB website: [www.hdb.gov.sg](https://www.hdb.gov.sg)
        - Data.gov.sg for HDB resale transaction data
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 4: Features
        st.markdown("<div class='about-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Features</h2>", unsafe_allow_html=True)
        
        features = [
            {
                "title": "HDB Resale Chatbot (Rina)",
                "description": [
                    "AI-powered assistant to answer user queries.",
                    "Utilizes natural language processing to understand and respond to questions.",
                    "Provides information based on the latest HDB guidelines and procedures."
                ]
            },
            {
                "title": "HDB Resale Flat Search",
                "description": [
                    "Interactive tool to search for resale flats within a specified budget.",
                    "Filters for town, flat type, and other preferences.",
                    "Displays relevant property details and transaction history."
                ]
            },
            {
                "title": "Data Visualizations",
                "description": [
                    "Price distribution charts.",
                    "Average price by town comparisons.",
                    "Price vs. floor area scatter plots."
                ]
            },
            {
                "title": "Methodology",
                "description": [
                    "Detailed explanation of our data processing and analysis methods.",
                    "Flowcharts illustrating the application's processes."
                ]
            }
        ]
    
        for feature in features:
            st.markdown(f"**{feature['title']}**:")
            for desc in feature["description"]:
                st.markdown(f"- {desc}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Methodology":
        # Custom CSS for styling the Methodology page
        st.markdown(
            """
            <style>
            .methodology-title {
                color: #2B2D42;
                font-size: 36px;
                text-align: center;
                margin-bottom: 20px;
            }
            .methodology-section {
                background-color: #ffffff; 
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Introductory Paragraph
        st.write("This section describes the data flow and implementation details of our HDB Resale Guide application. We use a combination of web scraping, data processing, and machine learning techniques to provide accurate and up-to-date information.")

        # Section 1: HDB Resale Chatbot
        st.markdown("<div class='methodology-section'>", unsafe_allow_html=True)
        st.markdown("<h2>1. HDB Resale Chatbot</h2>", unsafe_allow_html=True)
        st.write("""
        Our AI assistant, Rina, uses a retrieval-augmented generation (RAG) approach to answer user queries:
    
        1. User inputs a question about the HDB resale process.
        2. The query is passed to our Langchain model.
        3. Relevant documents are retrieved from:
            - Scraped HDB website content
            - Fallback JSON document (if web scraping yields no results)
        4. The retrieved context and user query are used to generate a response.
        5. The response is displayed to the user in the chat interface.
        6. User can provide feedback using thumbs up (positive) or thumbs down (negative) buttons for each response. 
            - For negative feedback, users can specify the type of issue (e.g. "Too brief", "Too vague", "Not relevant", "Incorrect information") and provide additional comments. 
        7. Chatbot responses will be dynamically modified according to user feedback. 
            - Based on the feedback, the system adjusts the prompt template used to generate responses. For example, if the feedback states that the response is "too brief", it adjusts the prompt to encourage more detailed responses. If the feedback states that the response is "too vague", it adjusts for more specific and concrete information.
        """)
    
        # Data Flow Diagram for HDB Resale Chatbot
        st.write("### Data Flow:")
        st.write("User Input → Query Processing → Document Retrieval → Context + Query to LLM → Response Generation → Display to User → Feedback Collection → Chatbot modifies responses based on feedback, if positive feedback ratio falls below 70%")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 2: HDB Resale Flat Search
        st.markdown("<div class='methodology-section'>", unsafe_allow_html=True)
        st.markdown("<h2>2. HDB Resale Flat Search</h2>", unsafe_allow_html=True)
        st.write("""
        Our flat search feature processes data as follows:
    
        1. User selects budget and preferences (town, flat type).
        2. Data is fetched from a CSV file containing recent HDB resale transactions.
        3. The application filters flats within the specified budget and preferences.
        4. Matching flats are displayed in a table format.
        5. Additional visualizations (price distribution, average price by town, etc.) are generated using Plotly.
        """)
    
        # Data Flow Diagram for HDB Resale Flat Search
        st.write("### Data Flow:")
        st.write("User Input (Budget/Preferences) → Data Filtering → Sorting → Result Display → Data Visualization")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 3: Implementation Details
        st.markdown("<div class='methodology-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Implementation Details</h2>", unsafe_allow_html=True)
        st.write("""
        - **Web Scraping**: BeautifulSoup is used to scrape the official HDB website daily. Target data includes policies, procedures, and FAQs. 
        - **Document Retrieval**: Based on a retrieval-augmented generation (RAG) approach. Relevant information is retrieved from: 
            1. Processed and indexed scraped HDB website content
            2. Fallback on JSON document (if web scraping yields no results)
        - **Data Processing**: Pandas is used for data cleaning, manipulation, and analysis. 
        - **Language Model**: Based on the GPT-4o-mini model, accessed via the Langchain library. This model uses the retrieved context and user query to formulate answers. 
        - **Visualization**: Plotly Express is used for creating interactive charts and graphs.
        - **Data Storage**: Scraped data is stored in CSV format for easy access and processing.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 4: Security Measures
        st.markdown("<div class='methodology-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Security Measures</h2>", unsafe_allow_html=True)
        st.write("""
        - User inputs are sanitized to prevent injection attacks.
        - The application is password-protected to control access.
        - Personal user data or chat histories will not be stored beyond the current session.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Section 5: Limitations and Future Improvements
        st.markdown("<div class='methodology-section'>", unsafe_allow_html=True)
        st.markdown("<h2>Limitations and Future Improvements</h2>", unsafe_allow_html=True)
        st.write("""
        - The chatbot's knowledge is limited to its training data and may not cover very recent changes.
        - The flat search feature relies on historical data and may not reflect real-time availability.
        - Future versions aim to incorporate real-time data feeds, implement predictive analytics for price forecasting, and expand the chatbot's capabilities to handle more complex queries and scenarios.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Flowchart for Use Cases
        st.write("## Flowchart for Use Cases:")
        st.image("https://raw.githubusercontent.com/jassyjazz/bootcampai/main/flowchart1.png", caption="HDB Resale Chatbot")
        st.image("https://raw.githubusercontent.com/jassyjazz/bootcampai/main/flowchart2.png", caption="HDB Resale Flat Search")

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

                # Display negative feedback form if needed
                if i in st.session_state.pending_feedback and not st.session_state.pending_feedback[i]['submitted']:
                    st.info("We're sorry to hear that. Please tell us what was wrong with the response:")
                    feedback_type = st.selectbox(
                        "What was the main issue?",
                        ["Too brief", "Too vague", "Not relevant", "Incorrect information", "Other"],
                        key=f"feedback_type_{i}"
                    )
                    detailed_feedback = st.text_area("Additional comments (optional):", key=f"detailed_feedback_{i}")
                    if st.button("Submit Feedback", key=f"submit_feedback_{i}"):
                        submit_negative_feedback(i)

        # Sidebar for feedback summary and analysis
        with st.sidebar:
            st.subheader("Feedback Summary")
            st.metric("Positive Feedback", st.session_state.feedback_count['positive'])
            st.metric("Negative Feedback", st.session_state.feedback_count['negative'])

            if st.button("View Detailed Analysis"):
                analyse_feedback()

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

                # Clear the existing traces and add new ones with custom hover text
                fig_area.data = []
                for flat_type in filtered_df['flat_type'].unique():
                    df_type = filtered_df[filtered_df['flat_type'] == flat_type]

                    hover_text = df_type.apply(
                        lambda row: f"Floor Area: {row['floor_area_sqm']} sqm<br>Price: ${row['resale_price']:,.0f}<br>Flat Type: {row['flat_type']}",
                        axis=1
                    )

                    fig_area.add_trace(
                        go.Scatter(
                            x=df_type['floor_area_sqm'],
                            y=df_type['resale_price'],
                            mode='markers',
                            name=flat_type,
                            text=hover_text,
                            hoverinfo='text',
                            marker=dict(size=8)
                        )
                    )

                st.plotly_chart(fig_area)

            else:
                st.write("No flats found within your budget and preferences.")
