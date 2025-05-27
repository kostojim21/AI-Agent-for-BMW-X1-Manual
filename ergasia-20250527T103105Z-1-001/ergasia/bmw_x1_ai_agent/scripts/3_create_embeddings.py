from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
import json
import os

with open(r"C:\Users\jimko\Desktop\ergasia\bmw_x1_ai_agent\scripts\output\chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = PersistentClient(path="output/vector_store")

collection = client.get_or_create_collection(name="bmw_manual")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk["text"]],
        metadatas=[chunk["metadata"]],
        ids=[f"chunk_{i}"]
    )

print("Embeddings αποθηκεύτηκαν στο vector store.")
