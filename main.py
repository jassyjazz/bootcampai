import streamlit as st
from langchain_community.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.output_parsers import RegexParser
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import os

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]

# Set page config
st.set_page_config(page_title="HDB Resale Flat Guide", layout="wide")

# Web scraping function
@st.cache_data(ttl=86400)  # Cache for 24 hours
def scrape_hdb_website():
    url = "https://www.hdb.gov.sg/residential/buying-a-flat/resale/getting-started"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract relevant information from the webpage
    # This is a simplified example; you'll need to adjust based on the actual structure of the HDB website
    content = soup.find('div', class_='content-area').get_text()
    return content

# Load JSON data
@st.cache_data
def load_json_data():
    with open('docs/buy_hdb_resale.json', 'r') as f:
        return json.load(f)

# Search function
def search_info(query):
    scraped_data = scrape_hdb_website()
    json_data = load_json_data()
    
    # Search in scraped data first
    if query.lower() in scraped_data.lower():
        return f"From HDB website: {scraped_data}"
    
    # If not found, search in JSON data
    for item in json_data:
        if query.lower() in item['question'].lower() or query.lower() in item['answer'].lower():
            return f"From our database: {item['answer']}"
    
    return "Sorry, I couldn't find relevant information for your query."

# Define the custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: list

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# Define the prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

# Define the tools
tools = [
    Tool(
        name="Search",
        func=search_info,
        description="useful for when you need to answer questions about HDB resale flat buying process"
    )
]

# Define the output parser
output_parser = RegexParser(
    regex=r"Action: (.*?)\nAction Input: (.*?)\nObservation: (.*?)\nThought: (.*?)\nFinal Answer: (.*)",
    output_keys=["action", "action_input", "observation", "thought", "final_answer"],
)

# Set up the agent
llm = OpenAI(temperature=0, model_name="gpt-4o-mini")
prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["input", "intermediate_steps"])

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Define the pages
def home():
    st.title("HDB Resale Flat Buying Guide")
    st.write("Welcome to your comprehensive guide for buying an HDB resale flat.")

def chat_guide():
    st.subheader("Chat with our AI Guide")
    user_input = st.text_input("Ask about buying an HDB resale flat:")
    if user_input:
        response = agent_executor.run(user_input)
        st.write(response)

    flat_type = st.selectbox("Select flat type:", ["3-room", "4-room", "5-room"])
    budget = st.slider("Your budget (SGD):", 200000, 1000000, 500000)
    
    if st.button("Get Personalized Advice"):
        query = f"I'm looking for a {flat_type} flat with a budget of SGD {budget}. What should I consider?"
        response = agent_executor.run(query)
        st.write(response)

def intelligent_search():
    st.subheader("Search HDB Resale Flat Information")
    search_query = st.text_input("Enter your search query:")
    if search_query:
        response = agent_executor.run(f"Search for information about: {search_query}")
        st.write(response)

    data = pd.DataFrame({
        'Flat Type': ['3-room', '4-room', '5-room'],
        'Average Price': [350000, 450000, 550000]
    })
    
    fig = px.bar(data, x='Flat Type', y='Average Price', title='Average HDB Resale Prices')
    st.plotly_chart(fig)

def about_us():
    st.subheader("Project Scope")
    st.write("This application aims to guide users through the process of buying an HDB resale flat in Singapore.")
    
    st.subheader("Objectives")
    st.write("1. Provide accurate and up-to-date information about HDB resale flat purchases")
    st.write("2. Offer personalized guidance based on user inputs")
    st.write("3. Enhance user understanding through interactive features")
    
    st.subheader("Data Sources")
    st.write("- Housing & Development Board (HDB) official website")
    st.write("- Singapore Government e-services")
    
    st.subheader("Features")
    st.write("1. Interactive Chat Guide")
    st.write("2. Intelligent Search")
    st.write("3. Data Visualizations")

def methodology():
    st.subheader("Data Flows and Implementation Details")
    st.write("Our application uses advanced NLP techniques and LLM models to process user queries and provide accurate information.")
    
    st.subheader("Process Flow: Chat Guide")
    st.image("chat_guide_flowchart.png")  # You need to create this image
    
    st.subheader("Process Flow: Intelligent Search")
    st.image("intelligent_search_flowchart.png")  # You need to create this image

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Chat Guide", "Intelligent Search", "About Us", "Methodology"])

# Display the selected page
if page == "Home":
    home()
elif page == "Chat Guide":
    chat_guide()
elif page == "Intelligent Search":
    intelligent_search()
elif page == "About Us":
    about_us()
elif page == "Methodology":
    methodology()
