import streamlit as st
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="AI PDF Summarizer")

st.title("📄 AI PDF Summarizer")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    # Combine all text
    full_text = ""

    for doc in docs:
        full_text += doc.page_content

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    # Prompt
    prompt = f"""
    Summarize the following PDF content in simple language:

    {full_text}
    """

    # Generate summary
    with st.spinner("Generating Summary..."):

        response = llm.invoke(prompt)

    # Show result
    st.subheader("Summary")

    st.write(response.content)
