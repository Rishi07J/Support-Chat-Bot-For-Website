import streamlit as st
from utils import *
import constants
import os

# ---------- Session State ----------
if 'HuggingFace_API_Key' not in st.session_state:
    st.session_state['HuggingFace_API_Key'] = ''
if 'Pinecone_API_Key' not in st.session_state:
    st.session_state['Pinecone_API_Key'] = ''

# ---------- Page Configuration ----------
st.set_page_config(page_title="AI Website Assistant", layout="wide")

# ---------- Main Title ----------
st.title("AI Assistance for Website")
st.markdown("Search and retrieve content from a website using semantic similarity with HuggingFace and Pinecone.")

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("API Configuration")

    st.session_state['HuggingFace_API_Key'] = st.text_input("HuggingFace API Key", type="password", placeholder="Enter HuggingFace API key")
    st.session_state['Pinecone_API_Key'] = st.text_input("Pinecone API Key", type="password", placeholder="Enter Pinecone API key")

    os.environ["PINECONE_API_KEY"] = st.session_state['Pinecone_API_Key']

    st.markdown("---")
    load_button = st.button("Load Website Data", key="load_button")

# ---------- Load Data ----------
if load_button:
    if st.session_state['HuggingFace_API_Key'] and st.session_state['Pinecone_API_Key']:
        with st.spinner("Fetching and processing website data..."):
            site_data = get_website_data(constants.WEBSITE_URL)
            st.success("Website data fetched.")

            chunks_data = split_data(site_data)
            st.success("Data split into chunks.")

            embeddings = create_embeddings()
            st.success("Embeddings created.")

            push_to_pinecone(
                st.session_state['Pinecone_API_Key'],
                constants.PINECONE_ENVIRONMENT,
                constants.PINECONE_INDEX,
                embeddings,
                chunks_data
            )
            st.sidebar.success("Data pushed to Pinecone successfully.")
    else:
        st.sidebar.error("Please enter both API keys before proceeding.")

# ---------- User Prompt ----------
st.subheader("Ask a Question About the Website")
prompt = st.text_input('Enter your query', key="prompt")
document_count = st.slider('Number of relevant results', 1, 5, 3, step=1)

submit = st.button("Search")

# ---------- Handle Search ----------
if submit:
    if st.session_state['HuggingFace_API_Key'] and st.session_state['Pinecone_API_Key']:
        with st.spinner("Searching for relevant documents..."):
            embeddings = create_embeddings()
            index = pull_from_pinecone(
                st.session_state['Pinecone_API_Key'],
                constants.PINECONE_ENVIRONMENT,
                constants.PINECONE_INDEX,
                embeddings
            )
            relevant_docs = get_similar_docs(index, prompt, document_count)
            st.success("Search completed.")

        # ---------- Display Results ----------
        st.subheader("Search Results")
        for i, doc in enumerate(relevant_docs, start=1):
            with st.expander(f"Result {i}"):
                st.markdown(f"**Content:**\n{doc.page_content}")
                st.markdown(f"**Source:** {doc.metadata['source']}")
    else:
        st.sidebar.error("Please enter both API keys before searching.")
