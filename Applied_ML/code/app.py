import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# --- Configuration & UI Setup ---
st.set_page_config(
    page_title="Healthcare Info Assistant",
    page_icon="⚕️",
    layout="centered"
)

load_dotenv()

# Verify API Key
if not os.getenv("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY in environment variables.")
    st.stop()

# Initialize Modern Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Resource Caching ---
@st.cache_resource
def load_embedding_model():
    # Matches the exact model used in ingest_medquad.py
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # Tries medquad_docs_csv first, falls back to medical_docs depending on your main collection
    try:
        return chroma_client.get_collection("medquad_docs_csv")
    except Exception:
        return chroma_client.get_collection("medical_docs")

embedding_model = load_embedding_model()
collection = load_collection()

# --- RAG Core Logic & System Engineering ---
def ask_healthcare_rag(question):
    # 1. Generate Query Embeddings via SentenceTransformer
    query_embedding = embedding_model.encode([question], convert_to_numpy=True).tolist()
    
    # 2. Retrieve Documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    
    # 3. Format Context and Compile Sources
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    context_blocks = []
    sources = set()
    
    for doc, meta in zip(documents, metadatas):
        # Extract metadata keys safely based on ingest variations
        source_name = meta.get("source", "Unknown Source")
        focus_area = meta.get("focus_area", "")
        focus_str = f" (Focus: {focus_area})" if focus_area and focus_area != "Unknown" else ""
        
        context_blocks.append(f"Document Source: {source_name}{focus_str}\nContent: {doc}")
        sources.add(source_name)
    
    context = "\n\n---\n\n".join(context_blocks)
    
    # 4. Prompt Engineering for Grounding, Safety, and Evaluation Metas
    system_prompt = f"""You are a trustworthy Healthcare Information Assistant. Your job is to answer the user's healthcare query based strictly on the provided medical context.

Follow these strict design constraints:
1. Grounded Response: Rely ONLY on the clear facts explicitly mentioned in the context. Do not extrapolate, assume, or use external knowledge.
2. Citations: Inside your response text, inline-cite the source name next to any claim or statement you make (e.g., "[Source: WHO Guidelines]").
3. Confidence & Uncertainty: At the absolute end of your response, provide a distinct section labeled "Confidence Estimation: [High / Medium / Low]" followed by a brief 1-sentence explanation of your certainty based on the evidence available. If the context is missing info or completely insufficient, explicitly state it and label confidence as Low.
4. Safety Layer: If the query requests emergency assistance, specific diagnosis of symptoms, or unsafe self-treatment advice, immediately refuse to answer and instruct the user to consult a licensed medical professional or local emergency unit.

Retrieved Context:
{context}

User Query:
{question}
"""

    # 5. Inference via Gemini 2.5 Flash
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=system_prompt
    )
    
    return response.text, list(sources)

# Helper function for text streaming effect
def stream_text(text):
    for word in text.split(" "):
        yield word + " "

# --- Main UI Layout ---
st.title("⚕️ Trustworthy Healthcare Info Assistant")
st.markdown(
    "Engage with a reliable information assistant grounded in trusted medical publications "
    "(MedQuAD, WHO, CDC, and NICE Guidelines)."
)

# Initialize Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Please present your health-related or lifestyle inquiry, and I will check our verified literature."
        }
    ]

# Render Active Thread
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Engagement Trigger
if prompt := st.chat_input("Ask about hypertension, diabetes, heart health, etc..."):
    
    # Log User query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Querying knowledge bases and checking evidence grounding..."):
            answer, sources = ask_healthcare_rag(prompt)
            
        # Stream structured answer output
        streamed_answer = st.write_stream(stream_text(answer))
        
        # Display explicit source attribution for metadata transparency
        st.markdown("---")
        st.caption("📚 **Retrieved Knowledge Source Base:** " + ", ".join(sources))
        
    # Store complete session text block
    st.session_state.messages.append({"role": "assistant", "content": streamed_answer})