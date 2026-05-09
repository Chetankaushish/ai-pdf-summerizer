import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pypdf import PdfReader

# Load Environment Variables
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

body {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #222;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
    font-size: 18px;
}

.box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

.stButton > button {
    width: 100%;
    height: 50px;
    background-color: #6C63FF;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #574bdb;
    color: white;
}

.summary-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="title">📄 AI PDF Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload PDF and generate AI summary instantly</div>',
    unsafe_allow_html=True
)

# Upload Box
st.markdown('<div class="box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

generate = st.button("✨ Generate Summary")

st.markdown('</div>', unsafe_allow_html=True)

# Generate Summary
if generate:

    if uploaded_file is None:

        st.warning("Please upload a PDF file.")

    else:

        try:

            # Read PDF
            pdf_reader = PdfReader(uploaded_file)

            text = ""

            for page in pdf_reader.pages[:5]:
                text += page.extract_text()

            text = text[:12000]

            # API Key
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                st.error("Google API Key not found!")
                st.stop()

            # Configure Gemini
            genai.configure(api_key=api_key)

            # Model
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Prompt
            prompt = f"""
            Summarize this PDF in simple language:

            {text}
            """

            # Generate Response
            with st.spinner("Generating Summary..."):

                response = model.generate_content(prompt)

            # Output
            st.markdown(
                '<div class="summary-box">',
                unsafe_allow_html=True
            )

            st.subheader("📌 Summary")

            st.write(response.text)

            # Download Button
            st.download_button(
                "⬇ Download Summary",
                response.text,
                file_name="summary.txt"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Error: {e}")
