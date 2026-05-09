import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import os
import time

# =========================
# Load Environment Variables
# =========================

load_dotenv()

# =========================
# OpenRouter Client
# =========================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="wide"
)

# =========================
# Sidebar Settings
# =========================

st.sidebar.title("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-2.0-flash-exp:free"
    ]
)

summary_length = st.sidebar.selectbox(
    "Summary Length",
    ["Short", "Medium", "Detailed"]
)

summary_language = st.sidebar.selectbox(
    "Summary Language",
    ["English", "Hindi"]
)

theme_mode = st.sidebar.radio(
    "Theme",
    ["Light", "Dark"]
)

# =========================
# Theme Colors
# =========================

if theme_mode == "Dark":

    background_color = "#0e1117"
    card_color = "#161b22"
    text_color = "white"

else:

    background_color = "#f5f7fb"
    card_color = "white"
    text_color = "#222"

# =========================
# Custom CSS
# =========================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {background_color};
    }}

    .main-title {{
        text-align: center;
        font-size: 58px;
        font-weight: 700;
        color: {text_color};
        margin-top: 20px;
    }}

    .sub-title {{
        text-align: center;
        color: gray;
        font-size: 20px;
        margin-bottom: 40px;
    }}

    .upload-container {{
        background-color: {card_color};
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
    }}

    .summary-container {{
        background-color: {card_color};
        padding: 30px;
        border-radius: 20px;
        margin-top: 25px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.08);
        color: {text_color};
    }}

    .stButton > button {{
        width: 100%;
        height: 55px;
        border-radius: 12px;
        background-color: #6C63FF;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }}

    .stButton > button:hover {{
        background-color: #574bdb;
        color: white;
    }}

    .footer {{
        text-align: center;
        margin-top: 40px;
        color: gray;
        font-size: 14px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header Section
# =========================

st.markdown(
    '<div class="main-title">📄 AI PDF Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload PDFs and generate AI-powered summaries instantly</div>',
    unsafe_allow_html=True
)

# =========================
# Upload Section
# =========================

st.markdown(
    '<div class="upload-container">',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "📂 Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

generate_summary = st.button("✨ Generate Summary")

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# =========================
# PDF Text Extraction
# =========================

def extract_pdf_text(files):

    combined_text = ""

    for uploaded_file in files:

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages[:3]:

            page_text = page.extract_text()

            if page_text:
                combined_text += page_text

    return combined_text[:6000]

# =========================
# Prompt Builder
# =========================

def create_prompt(text, language, length):

    if language == "Hindi":
        language_instruction = "Generate the summary in Hindi."
    else:
        language_instruction = "Generate the summary in English."

    if length == "Short":
        length_instruction = "Provide a short summary using bullet points."

    elif length == "Medium":
        length_instruction = "Provide a medium-length clear summary."

    else:
        length_instruction = "Provide a detailed summary with explanations."

    prompt = f"""
    {language_instruction}

    {length_instruction}

    Summarize the following PDF content:

    {text}
    """

    return prompt

# =========================
# Generate AI Summary
# =========================

if generate_summary:

    if not uploaded_files:

        st.warning("⚠️ Please upload at least one PDF file.")

    else:

        try:

            # Extract Text
            pdf_text = extract_pdf_text(uploaded_files)

            # Build Prompt
            final_prompt = create_prompt(
                pdf_text,
                summary_language,
                summary_length
            )

            # Generate Response
            with st.spinner("Generating AI Summary..."):

                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ]
                )

            summary = response.choices[0].message.content

            # =========================
            # Summary Output
            # =========================

            st.markdown(
                '<div class="summary-container">',
                unsafe_allow_html=True
            )

            st.subheader("📌 AI Generated Summary")

            typing_placeholder = st.empty()

            displayed_text = ""

            for word in summary.split():

                displayed_text += word + " "

                typing_placeholder.markdown(displayed_text)

                time.sleep(0.02)

            # Download Button
            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(f"❌ Error: {error}")

# =========================
# Footer
# =========================

st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit + OpenRouter</div>',
    unsafe_allow_html=True
)
