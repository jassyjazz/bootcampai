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

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ... (rest of the imports and functions remain the same)

# Modified HDB Resale Flat Search page
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
            filtered_df['resale_price'] = filtered_df['resale_price'].apply(lambda x: f"${x:,.0f}")
            filtered_df['month'] = filtered_df['month'].dt.strftime('%Y-%m')
            columns_to_display = ['Town', 'Flat Type', 'Block', 'Street Name', 'Storey Range', 'Floor Area Sqm', 'Remaining Lease', 'Resale Price', 'Month']
            filtered_df.columns = filtered_df.columns.str.title().str.replace('_', ' ')
            st.dataframe(filtered_df[columns_to_display].reset_index(drop=True))

            # Visualizations
            st.subheader("Visualizations")

            # Price Distribution
            fig_price = px.histogram(filtered_df, x="Resale Price", title="Price Distribution of Matching Flats")
            st.plotly_chart(fig_price)

            # Average Price by Town
            avg_price_by_town = filtered_df.groupby("Town")["Resale Price"].agg(lambda x: x.str.replace('$', '').str.replace(',', '').astype(float).mean())
            avg_price_by_town = avg_price_by_town.sort_values(ascending=False)
            fig_town = px.bar(avg_price_by_town, x=avg_price_by_town.index, y="Resale Price", title="Average Price by Town")
            st.plotly_chart(fig_town)

            # Price vs Floor Area
            filtered_df['Resale Price'] = filtered_df['Resale Price'].str.replace('$', '').str.replace(',', '').astype(float)
            fig_area = px.scatter(filtered_df, x="Floor Area Sqm", y="Resale Price", color="Flat Type", title="Price vs Floor Area")
            st.plotly_chart(fig_area)

        else:
            st.write("No flats found within your budget and preferences.")

# Modified HDB Resale Chatbot page
elif page == "HDB Resale Chatbot":
    st.write("""
        Hi! I am **Rina**, your virtual HDB assistant. I'm here to help you with any questions you have about the HDB resale process.
        Whether you're wondering about **eligibility**, **how to apply for a resale flat**, or **the procedures involved**, just ask me anything.
        I'm here to guide you every step of the way. Simply type your question below, or click on one of the common queries to get started.
    """)

    # Common queries as buttons
    common_queries = [
        "What are the eligibility criteria for buying an HDB resale flat?",
        "How do I apply for an HDB resale flat?",
        "What is the HDB resale process?",
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
            sanitized_question = user_question.replace("{", "").replace("}", "").replace("\\", "")
            st.session_state.chat_history.append(("Human", sanitized_question))

            with st.spinner("Generating response..."):
                response = chain.invoke(sanitized_question)
            st.session_state.chat_history.append(("Rina", response))
        else:
            st.write("Please enter a question to get started.")

    # Display chat history with improved formatting
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "Human":
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>You:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Rina:</strong> {message}</div>", unsafe_allow_html=True)

        if i < len(st.session_state.chat_history) - 1:
            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
