# rag_chatbot.py

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def build_faiss_vector_store():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found in .env file")

    # 1. Load CSV
    df = pd.read_csv("angelone_faqs.csv")
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['Question']}\nA: {row['Answer']}"
        docs.append(Document(page_content=content))

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Create embeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # 4. Build vector store
    db = FAISS.from_documents(chunks, embedding)
    db.save_local("faiss_db")
    print("✅ Vector database created and saved to /faiss_db")
