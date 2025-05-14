import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ====== Groq API Setup ======
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please define it as an environment variable.")

client = Groq(api_key=GROQ_API_KEY)


# ====== Load Cases ======
with open("detailed_summarized_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

# Create TF-IDF search index
texts = [f"{case['case_name']} {case['question']} {case['answer']}" for case in cases]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# ====== UI ======
st.set_page_config(page_title="JUDICIO", layout="centered")
st.markdown("### 🏛️ JUDICIO")
st.markdown("Ask questions about Indian legal cases, IPC sections, or legal concepts. You'll get relevant judgments and AI-powered responses.")

# Add a selector for query type
query_type = st.radio(
    "Select query type:",
    ["Case Search", "IPC Section", "Legal Concept"]
)

user_query = st.text_input("Enter your legal question:")

# Display area for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Search + Groq Response Function ======
def legal_chatbot_response(query, query_type):
    # Top similar cases
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = scores.argsort()[-3:][::-1]

    response_text = ""
    combined_context = ""

    for idx in top_indices:
        case = cases[idx]
        context = f"Case: {case['case_name']} ({case.get('judgement_date', 'N/A')})\nQ: {case['question']}\nA: {case['answer']}\n"
        combined_context += context
        response_text += f"**{case['case_name']}** ({case.get('judgement_date', 'N/A')})\n- {case['answer']}\n\n"

    # prompt based on query type
    if query_type == "Case Search":
        system_prompt = "You are a top-notch legal assistant specializing in Indian case law. Answer based on the provided case summaries."
        user_prompt = f"Use the following legal case summaries to answer the user's question:\n\n{combined_context}\n\nUser's question: {query}\n\nAnswer clearly and accurately based on the above case summaries."
    
    elif query_type == "IPC Section":
        system_prompt = "You are a legal expert specializing in the Indian Penal Code (IPC). Provide detailed explanations of IPC sections, their applications, and relevant case law."
        user_prompt = f"The user is asking about an IPC section: '{query}'\n\nHere are some potentially relevant cases:\n\n{combined_context}\n\nExplain the IPC section, its key provisions, penalties, and how it has been interpreted in the cases provided or other landmark judgments. Be comprehensive and cite specific sections and subsections."
    
    else:  # Legal Concept
        system_prompt = "You are a legal scholar specializing in Indian jurisprudence and legal concepts. Explain legal principles, doctrines, and their applications in Indian law."
        user_prompt = f"The user is asking about a legal concept: '{query}'\n\nExplain this legal concept thoroughly, its origins, how it's applied in Indian law, and cite relevant statutory provisions and case law examples."

    groq_reply = "AI reply unavailable."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",  
            temperature=0.3,  
            max_completion_tokens=1024,  
            top_p=1
        )
        groq_reply = chat_completion.choices[0].message.content
    except Exception as e:
        groq_reply = f"Error: {str(e)}"

    return response_text, groq_reply

# ====== Triggered on Submit ======
if user_query:
    similar_cases, groq_answer = legal_chatbot_response(user_query, query_type)

    # Only show relevant cases for Case Search and IPC Section
    if query_type in ["Case Search", "IPC Section"]:
        st.markdown("### 🔍 Relevant Legal Cases")
        st.markdown(similar_cases)

    st.markdown(f"LAW ASSISTANT ({query_type})")
    st.success(groq_answer)

    st.session_state.chat_history.append((user_query, groq_answer))

# ====== Show Chat History ======
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
