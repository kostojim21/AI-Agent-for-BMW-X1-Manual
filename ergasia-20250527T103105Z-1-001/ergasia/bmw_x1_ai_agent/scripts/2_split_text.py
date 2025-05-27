from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

with open(r"C:\Users\jimko\Desktop\ergasia\manual_text_output.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_text(full_text)

docs = [
    Document(page_content=chunk, metadata={"source": f"Σελίδα ~{i+1}"})
    for i, chunk in enumerate(chunks)
]

json_chunks = [
    {"text": doc.page_content, "metadata": doc.metadata}
    for doc in docs
]

with open(r"C:\Users\jimko\Desktop\ergasia\bmw_x1_ai_agent\scripts\output\chunks.json", "w", encoding="utf-8") as f:
    json.dump(json_chunks, f, ensure_ascii=False, indent=2)

print(f"Δημιουργήθηκαν {len(docs)} τα chunks κ αποθηκεύτηκαν στο output/chunks.json")
