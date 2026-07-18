import os
import time
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Define file assets
documents = [
    "data/CDC/Living with Diabetes.pdf",
    "data/CDC/Preventing Heart Disease.pdf",
    "data/NICE/Hypertension in adults diagnosis and management.pdf",
    "data/WHO/Management of Advanced HIV Disease.pdf",
    "data/WHO/Supporting healthy diets and safe foods in urban food environments.pdf"
]

# 2. Configure Environment & API Keys
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from your .env file!")

# 3. Setup ChromaDB Embedding Function
google_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)

# 4. Initialize Database and Reset Collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection("medical_docs")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="medical_docs", 
    embedding_function=google_ef
)

# 5. Extract Text and Chunk Data
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []
all_ids = []
all_metadatas = []
chunk_id = 0

for pdf_file in documents:
    print(f"Processing: {pdf_file}")
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    chunks = splitter.split_text(text)

    # Use os.path.basename to handle Windows and Linux paths correctly
    source_filename = os.path.basename(pdf_file)

    for chunk in chunks:
        all_chunks.append(chunk)
        all_ids.append(str(chunk_id))
        all_metadatas.append({"source": source_filename})
        chunk_id += 1

# 6.Ingestion with Throttled Batching for Free Tier
# Using a smaller batch size and longer sleep to stay safely under 100 RPM
BATCH_SIZE = 15
total_items = len(all_chunks)

print(f"\nStarting ingestion of {total_items} total chunks...")

for i in range(0, total_items, BATCH_SIZE):
    batch_docs = all_chunks[i : i + BATCH_SIZE]
    batch_ids = all_ids[i : i + BATCH_SIZE]
    batch_meta = all_metadatas[i : i + BATCH_SIZE]

    print(f" -> Upserting batch {i // BATCH_SIZE + 1}/{(total_items + BATCH_SIZE - 1) // BATCH_SIZE} (Items {i} to {i + len(batch_docs)})")
    
    # Wrap in a retry mechanism in case the API limit is still recovering
    while True:
        try:
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta,
            )
            break  # Success, break the retry loop
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("    Rate limit reached. Backing off for 30 seconds...")
                time.sleep(30)
            else:
                raise e  # Re-raise other unexpected errors

    # Pause between successful batches to respect the 100 RPM ceiling
    if i + BATCH_SIZE < total_items:
        print("    Waiting 10 seconds to respect free-tier rate limits...")
        time.sleep(10)

print("\nIngestion Completed successfully!")

