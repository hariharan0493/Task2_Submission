import os
import time
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/medquad.csv")
print(df.head())

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# try:
#     chroma_client.delete_collection("medquad_docs")
# except Exception:
#     pass

collection = chroma_client.get_or_create_collection(
    name="medquad_docs_csv"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = []
metadatas = []
ids = []
id_counter = 0

print("Chunking data...")
for index, row in df.iterrows():
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    source = str(row.get("source", "Unknown"))
    focus_area = str(row.get("focus_area", "Unknown"))

    if not question and not answer:
        continue

    text = f"Question: {question}\nAnswer: {answer}"
    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append(
            {
                "source": source,
                "focus_area": focus_area,
                "chunk_id": i,
            }
        )
        ids.append(str(id_counter))
        id_counter += 1

print(f"Total chunks to ingest: {len(documents)}")

BATCH_SIZE = 40
start = collection.count()

print(start, "number of chunks already in the collection")
print("Generating embeddings and ingesting in batches...")

for i in range(start, len(documents), BATCH_SIZE):

    batch_docs = documents[i:i + BATCH_SIZE]
    batch_meta = metadatas[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]

    embeddings = embedding_model.encode(
        batch_docs,
        convert_to_numpy=True,
        show_progress_bar=False
    ).tolist()

    collection.add(
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=batch_meta,
        ids=batch_ids
    )

    current_batch = i // BATCH_SIZE + 1
    total_batches = (len(documents) // BATCH_SIZE) + 1

    print(
        f"Batch {current_batch}/{total_batches} "
        f"| Documents: {len(batch_docs)} "
        f"| Status: Success"
    )

print("\n")
print("MedQuAD ingestion completed successfully.")
print(f"Total chunks stored : {len(documents)}")