# NLP-based AI-Powered Indian Legal Case Explorer

This project is a web-based legal information assistant that helps users search and understand Indian legal cases, IPC sections, and legal concepts using natural language queries.

The system combines semantic similarity search with an LLM-based legal assistant to provide relevant case summaries and contextual legal explanations.

## Features
- Search and retrieve relevant Indian legal case summaries using TF-IDF and cosine similarity
- Supports three query types: Case Search, IPC Section and Legal Concept
- AI-powered legal explanations using Groq LLM
- Interactive web interface built with Streamlit

## Tech Stack
- Python
- Streamlit
- Scikit-learn (TF-IDF, cosine similarity)
- Groq API (LLM)

## Dataset
The application uses a pre-processed JSON dataset containing summarized Indian legal cases and judgments.

## How to Run

1. Create and activate a virtual environment.
2. Install dependencies.
3. Set the Groq API key as an environment variable:
   ```bash
   export GROQ_API_KEY=your_api_key
