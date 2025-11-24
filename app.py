import streamlit as st
import PyPDF2
import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# -----------------------------------------
# 1. PDF Extraction Function
# -----------------------------------------
def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# -----------------------------------------
# 2. Prompt Template (LangChain)
# -----------------------------------------
resume_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Extract structured resume information in VALID JSON only.

Required JSON structure:
{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "skills": [],
  "education": [],
  "experience": [],
  "summary": ""
}}

Resume Text:
{resume_text}
"""
)

# -----------------------------------------
# 3. Helper – Clean JSON reply
# -----------------------------------------
def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0)) if match else {"error": "Invalid JSON"}


# -----------------------------------------
# 4. Streamlit UI
# -----------------------------------------
st.title("📄 Resume Parser using Groq + LangChain")
st.write("Upload a PDF resume and extract structured JSON data.")

# Input: API Key
api_key = st.text_input("🔑 Enter your Groq API Key", type="password")

# Input: PDF Upload
uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

# Button
if st.button("Parse Resume"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not uploaded_file:
        st.error("Please upload a PDF resume.")
    else:
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            resume_text = extract_pdf_text(uploaded_file)

        # Build LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        chain = resume_prompt | llm

        with st.spinner("Analyzing resume using Groq LLM..."):
            response = chain.invoke({"resume_text": resume_text})
            data = extract_json(response.content)

        st.success("Resume parsed successfully!")
        st.json(data)
