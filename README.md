# 📄 AI Resume Analyzer & Job Matcher

An intelligent recruitment tool that uses **Large Language Models (LLMs)** to evaluate the compatibility between a candidate's resume and a job description. This project leverages the high-speed **Groq Llama-3** models to provide near-instant, precise match percentages.

---

## 🚀 Key Features

* **Multi-Model Analysis:** Supports benchmarking across Llama 3.3 70B, Llama 3.1 8B, and Mixtral via Groq.
* **Production-Ready Extraction:** Uses robust Regular Expressions (Regex) to ensure clean integer outputs from LLM responses.
* **Optimized Prompting:** Implements a structured scoring rubric (Hard Skills, Experience, Industry Fit).
* **Shadowing-Safe Code:** Built-in protection against Python name-clashing and global variable errors.

---

## 🛠️ Technical Stack

* **Language:** Python 3.10+
* **Core API:** [Groq Cloud](https://console.groq.com/)
* **Models:** `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`
* **Libraries:** `re`, `groq`, `langchain-groq` (optional),`OpenAI`
* **Deployment:** Environment-variable driven for secure API key management.

---

## ⚙️ Installation

**Clone the Repository:**

    git clone https://github.com/Ahmed2797/Resume-Agent
    cd Resume-Agent

    ## Install Dependencies

    pip install groq python-dotenv
    ## Environment Setup

    GROQ_API_KEY=your_gsk_api_key_here
    DEEPSEEK_API_KEY=your_gsk_api_key_here
    GOOGLE_API_KEY=your_gsk_api_key_here
    ANTHROPIC_API_KEY=your_gsk_api_key_here

## 💡 Usage Example

    from resume.helper import get_groq_match

    # Define your texts
    resume = "Machine Learning Engineer with PyTorch & MLOps experience..."
    jd = "Looking for an AI Engineer proficient in Python and AWS..."

    # Generate prompt and get score
    prompt = build_prompt(resume, jd)
    match_score = get_groq_match(prompt)

    print(f"Match Confidence: {match_score}%")

    ## run code 
    python resume_agent.py
    streamlit run resume_agent.py
