import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import unicodedata
import warnings
import logging
from deep_translator import GoogleTranslator  

genai.configure(api_key="AIzaSyCH_aCbyn0NR4BgoKJxXkbkM9uwARI7ZRw")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="output/vector_store")
collection = client.get_collection("bmw_manual")

def call_gemini(prompt):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=genai.GenerationConfig(
                temperature=1
            )
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Σφάλμα κατά την κλήση Gemini: {str(e)} ⚠️"

def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_greek(text):
    return GoogleTranslator(source='auto', target='el').translate(text)

def process_question(question, top_k=3):
    try:
        question_en = translate_to_english(question)

        embedding = embedding_model.encode(question_en).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=top_k)

        documents = results['documents'][0]
        metadata = results['metadatas'][0]

        if not documents:
            return "❌ Δεν βρέθηκε σχετική πληροφορία. ❌"

        context = ""
        for doc, meta in zip(documents, metadata):
            source = meta.get("source", "Άγνωστη σελίδα")
            context += f"\n[Πηγή: {source}]\n{doc}\n"

        prompt = f"""Answer the following question based on the excerpts from the BMW X1 manual analytically.

Question: {question_en}

Excerpts:
{context}

Answer:"""

        answer_en = call_gemini(prompt)
        return translate_to_greek(answer_en)

    except Exception as e:
        return f"⚠️ Σφάλμα επεξεργασίας: {str(e)} ⚠️"

if __name__ == "__main__":
    print("🚗 Βοηθός BMW Manual – γράψτε την ερώτησή σας ή 'exit' για έξοδο.")
    while True:
        question = input("❓ Ερώτηση: ")
        if question.strip().lower() in ["exit", "quit", "έξοδος"]:
            print("👋 Έξοδος...")
            break
        response = process_question(question)
        print("🤖 Απάντηση:\n", response)
