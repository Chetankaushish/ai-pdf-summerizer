import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Environment Variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #222;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.upload-container {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
}

.summary-container {
    background: white;
    padding: 25px;
    border-radius: 15px;
    margin-top: 25px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
}

.stButton > button {
    width: 100%;
    height: 52px;
    background-color: #6C63FF;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #5548e5;
    color: white;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="title">📄 AI PDF Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your PDF and generate AI-powered summaries instantly</div>',
    unsafe_allow_html=True
)

# ---------------- UPLOAD SECTION ---------------- #

st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload Your PDF File",
    type=["pdf"]
)

generate_summary = st.button("✨ Generate Summary")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN LOGIC ---------------- #

if generate_summary:

    if uploaded_file is None:

        st.warning("⚠ Please upload a PDF file.")

    else:

        try:

            # Save Uploaded PDF
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )

            docs = text_splitter.split_documents(documents)

            # Extract Limited Text
            full_text = ""

            for doc in docs[:5]:
                full_text += doc.page_content

            # Limit Characters
            full_text = full_text[:12000]

            # API Key
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:

                st.error("❌ Google API Key not found!")
                st.stop()

            # Configure Gemini
            genai.configure(api_key=api_key)

            # Gemini Model
            model = genai.GenerativeModel("models/gemini-1.5-flash")

            # Prompt
            prompt = f"""
            Summarize the following PDF content in simple and clear language.

            PDF Content:
            {full_text}
            """

            # Generate Summary
            with st.spinner("⏳ Generating Summary..."):

                response = model.generate_content(prompt)

            # Display Summary
            st.markdown(
                '<div class="summary-container">',
                unsafe_allow_html=True
            )

            st.subheader("📌 Summary")

            st.write(response.text)

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"❌ Error: {e}")

# ---------------- FOOTER ---------------- #

st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit + Gemini AI</div>',
    unsafe_allow_html=True
)
