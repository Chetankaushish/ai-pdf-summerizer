import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #262730;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #574bdb;
    color: white;
}

.upload-box {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

.summary-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📄 AI PDF Summarizer</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Upload your PDF and generate AI summary instantly</div>',
    unsafe_allow_html=True
)

# Upload Section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

generate = st.button("✨ Generate Summary")

st.markdown('</div>', unsafe_allow_html=True)

# Main Logic
if generate:

    if uploaded_file is None:
        st.warning("Please upload a PDF file.")

    else:

        try:

            # Save PDF
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(documents)

            # Collect text
            full_text = ""

            for doc in docs[:5]:
                full_text += doc.page_content

            full_text = full_text[:12000]

            # API KEY
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                st.error("Google API Key not found!")
                st.stop()

            # Gemini SDK
            import google.generativeai as genai

            genai.configure(api_key=api_key)

            model = genai.GenerativeModel("gemini-1.5-flash")

            # Prompt
            prompt = f"""
            Summarize this PDF in simple language:

            {full_text}
            """

            # Generate Summary
            with st.spinner("Generating Summary..."):

                response = model.generate_content(prompt)

            # Show Summary
            st.subheader("📌 Summary")

            st.write(response.text)

        except Exception as e:

            st.error(f"Error: {e}")
