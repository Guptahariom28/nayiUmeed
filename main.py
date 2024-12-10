import streamlit as st
from wikipedia_retrieval import tool_wikipedia
from arxiv_retrieval import tool_arxiv
from faiss_database import load_faiss_data, get_response_from_faiss, add_query_to_faiss, save_faiss_data
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import HuggingFaceHub

# Load FAISS data
load_faiss_data()

# Initialize LLM
llm = HuggingFaceHub(model_name="meta-llama/Llama-2-7b-chat-hf", api_key="YOUR_HF_API_KEY")

# Combine tools
tools = [
    Tool(name="Wikipedia", func=tool_wikipedia.run, description="Search Wikipedia."),
    Tool(name="Arxiv", func=tool_arxiv.run, description="Search Arxiv academic papers.")
]

# Initialize LangChain agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit interface
st.title("Information Retrieval App with FAISS")
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        # Check FAISS cache
        cached_response = get_response_from_faiss(query)
        if cached_response:
            st.write("**Cached Response:**")
            st.write(cached_response)
        else:
            # Query the agent
            response = agent.run(query)
            st.write("**Agent Response:**")
            st.write(response)

            # Add to FAISS cache
            add_query_to_faiss(query, response)

        # Save FAISS data
        save_faiss_data()
