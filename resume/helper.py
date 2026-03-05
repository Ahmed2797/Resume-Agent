import os
import re
import openai
import pdfplumber
from io import StringIO
import streamlit as st
from openai import OpenAI
# from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader
)
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

openai = OpenAI(api_key=openai_api_key)

def load_and_split_resume(file_path: str):
    """
    Loads a resume file and splits it into text chunks using LangChain.

    Args:
        file_path (str): Path to the resume file (.txt, .pdf, .docx, etc.)
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap between chunks to preserve context.

    Returns:
        List[str]: List of split text chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # Select the appropriate loader
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        # Fallback for other common formats
        loader = UnstructuredFileLoader(file_path)

    # Load the file as LangChain documents
    documents = loader.load()

   
    return documents
    # return [doc.page_content for doc in split_docs]

# Function to extract text from uploaded files
def extract_text(file):
    """
    Extract text from uploaded file.
    Supports PDF and TXT files.
    Returns extracted text as string.
    """

    try:
        # ----------- PDF FILE -----------
        if file.name.lower().endswith(".pdf"):
            texts = []

            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)

            return "\n".join(texts)

        # ----------- TEXT FILE -----------
        elif file.name.lower().endswith(".txt"):
            content = file.read()

            # If content is bytes, decode it
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

            return content

        # ----------- UNSUPPORTED FORMAT -----------
        else:
            return "Unsupported file format. Please upload PDF or TXT file."

    except Exception as e:
        return f"Error extracting text: {str(e)}"

def get_groq_client(model_name="llama-3.3-70b-versatile"):
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model_name,
        temperature=0
    )

def extract_score(ai_response):
    """Helper to pull the integer out of the response object"""
    content = ai_response.content
    match = re.search(r'\d+', content)
    return min(int(match.group()), 100) if match else 0

# Function to get match percentage from OpenAI GPT-4
def get_openai_match(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        match = re.search(r'\d+', content)
        percentage = int(match.group()) if match else 0
        return min(percentage, 100)
        # return extract_score(content)
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return 0

def get_groq_match(prompt):
    try:
        # Mapping "OpenAI-level" quality to Groq's best model
        llm = get_groq_client("llama-3.3-70b-versatile")
        response = llm.invoke(prompt)
        return extract_score(response)
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return 0

def get_anthropic_match(prompt):
    try:
        # Mapping "Anthropic-level"
        llm = get_groq_client("llama-3.1-8b-instant")
        response = llm.invoke(prompt)
        return extract_score(response)
    except Exception as e:
        st.error(f"Anthropic API Error: {e}")
        return 0
    
def get_google_match(prompt):
    try:    
        # Mapping "Gemini-level" to Mixtral
        llm = get_groq_client("openai/gpt-oss-120b")
        response = llm.invoke(prompt)
        return extract_score(response)
    except Exception as e:
        st.error(f"Google Gemini API Error: {e}")
        return 0

def get_deepseek_match(prompt):
    try:
        llm = get_groq_client("groq/compound-mini")
        response = llm.invoke(prompt)
        return extract_score(response)
    except Exception as e:
        st.error(f"DeepSeek API Error: {e}")
        return 0



## use api and this helper function in resume_agent.py to get the match percentage from different models and display them in the UI.

import streamlit as st
import os
from openai import OpenAI
import re
# from anthropic import Anthropic

# Function to get match percentage from OpenAI GPT-4
def get_openai_match_key(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional resume evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        # digits = ''.join(filter(str.isdigit, content))
        # return min(int(digits), 100) if digits else 0

        match = re.search(r'\d+', content)
        percentage = int(match.group()) if match else 0
        return min(percentage, 100)
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return 0

# Function to get match percentage from Anthropic Claude
def get_anthropic_match_key(prompt):
    try:
        model_name = "claude-3-7-sonnet-latest"
        claude = 'pip install Anthropic' # Anthropic()
        message = claude.messages.create(
            model=model_name,
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = message.content[0].text
        digits = ''.join(filter(str.isdigit, content))
        return min(int(digits), 100) if digits else 0
    except Exception as e:
        st.error(f"Anthropic API Error: {e}")
        return 0

# Function to get match percentage from Google Gemini
def get_google_match_key(prompt):
    try:
        gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        model_name = "gemini-2.0-flash"
        messages = [{"role": "user", "content": prompt}]
        response = gemini.chat.completions.create(model=model_name, messages=messages)
        content = response.choices[0].message.content
        digits = ''.join(filter(str.isdigit, content))
        return min(int(digits), 100) if digits else 0
    except Exception as e:
        st.error(f"Google Gemini API Error: {e}")
        return 0

# Function to get match percentage from Groq
def get_groq_match_key(prompt):
    try:
        groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
        model_name = "llama-3.3-70b-versatile"
        messages = [{"role": "user", "content": prompt}]
        response = groq.chat.completions.create(model=model_name, messages=messages)
        answer = response.choices[0].message.content
        digits = ''.join(filter(str.isdigit, answer))
        return min(int(digits), 100) if digits else 0
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return 0

# Function to get match percentage from DeepSeek
def get_deepseek_match_key(prompt):
    try:
        deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
        model_name = "deepseek-chat"
        messages = [{"role": "user", "content": prompt}]
        response = deepseek.chat.completions.create(model=model_name, messages=messages)
        answer = response.choices[0].message.content
        digits = ''.join(filter(str.isdigit, answer))
        return min(int(digits), 100) if digits else 0
    except Exception as e:
        st.error(f"DeepSeek API Error: {e}")
        return 0
