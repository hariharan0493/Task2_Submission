# Design Document: Healthcare Information Assistant

## 1. Overview
The Healthcare Information Assistant is a trustworthy retrieval-augmented generation (RAG) system designed to answer medical queries based on evidence from authoritative sources. The system prioritizes grounding, citation accuracy, and safety over conversational flexibility.

## 2. Architecture & Ingestion Strategy
The system follows a modular pipeline utilizing a dual-embedding strategy to balance semantic quality and processing efficiency:
*   **PDF Ingestion (High-Fidelity)**: For complex medical guidelines (e.g., CDC, NICE, WHO), the system uses Google's `gemini-embedding-001` to capture deep semantic meaning. To successfully process these within free-tier API limits, `time.sleep` methods were integrated into the ingestion script to pace the requests.
*   **MedQuAD Ingestion (Local Efficiency)**: The extensive MedQuAD CSV dataset is embedded locally using `SentenceTransformer("all-MiniLM-L6-v2")`. This allows for rapid, offline vectorization, which is then batch-inserted (in chunks of 40) into the database.
*   **Text Chunking**: All ingested text is processed using a `RecursiveCharacterTextSplitter` configured with a chunk size of 1000 characters and a 200-character overlap. This specific overlap ensures that context isn't lost at the boundaries of large paragraphs.
*   **Vector Database**: All embeddings, text chunks, and source metadata are stored persistently in ChromaDB collections (`medical_docs` and `medquad_docs_csv`) for fast retrieval.

## 3. Prompt Engineering & Hallucination Prevention
To guarantee safety and prevent the AI from generating unsupported claims, the system relies on a highly structured prompt passed to the Gemini 3.5 Flash model. The prompt enforces the following core behaviors:
*   **Strict Grounding**: The LLM is explicitly instructed to rely *only* on the retrieved context blocks. It is forbidden from extrapolating, assuming, or bringing in external training data.
*   **Inline Citations**: The prompt mandates that every factual claim must be immediately followed by an inline citation referencing the specific document metadata (e.g., "[Source: WHO Guidelines]").
*   **Explicit Confidence Estimation**: The model must append a distinct "Confidence Estimation: [High / Medium / Low]" section at the very end of its response, backed by a one-sentence explanation. If the retrieved context is insufficient to answer the query, the model is instructed to explicitly state this and default to a Low confidence score.
*   **Safety Layer**: A strict directive acts as a safety guardrail; if a user requests emergency assistance, a specific diagnosis, or unsafe self-treatment, the LLM is programmed to immediately refuse the query and redirect the user to a licensed medical professional.

## 4. UI/UX Design Choices
*   **Framework**: The frontend is built with Streamlit, utilizing its session state to maintain an active chat history.
*   **Streaming Responses**: To improve perceived performance and user experience, the LLM's output is yielded word-by-word using a custom text streaming function.
*   **Metadata Transparency**: Regardless of the generated text, the UI explicitly renders a "Retrieved Knowledge Source Base" caption beneath every answer, listing the exact files (e.g., MedQuAD, Living with Diabetes) pulled from the database for that interaction.
