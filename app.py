import streamlit as st
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain

# Load environment variables
load_dotenv()

# Streamlit page settings
st.set_page_config(page_title="AI PDF Summarizer")

st.title("📄 AI PDF Summarizer")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    # Load Gemini model
    llm = ChatGoogleGenerativeAI(
        model="model/gemini-1.5-flash",
        temperature=0.3
    )

    # Create summarize chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )

    # Generate summary
    with st.spinner("Generating Summary..."):
        summary = chain.run(docs)

    # Show summary
    st.subheader("Summary")
    st.write(summary)
