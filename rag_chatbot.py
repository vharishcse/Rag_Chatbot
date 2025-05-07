# rag_chatbot.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Load API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in .env file")

# 2. Load CSV and convert to Document objects
def load_faq_csv(path):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['Question']}\nA: {row['Answer']}"
        docs.append(Document(page_content=content))
    return docs

docs = load_faq_csv("angelone_faqs.csv")

# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 4. Embeddings & vector store
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_db = FAISS.from_documents(chunks, embedding)
vector_db.save_local("faiss_db")

# 5. Reload DB
db = FAISS.load_local("faiss_db", embedding, allow_dangerous_deserialization=True)

# 6. Setup LLM + QA chain
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=api_key)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and accurate AngelOne support assistant.
Use ONLY the below context to answer the question. 
If the answer is not explicitly present or relevant, reply: "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# 7. Generate response from user query
def generate_response(user_query):
    relevant_docs = db.similarity_search(user_query, k=4)
    if not relevant_docs:
        return "I don't know"
    response = qa_chain.run(input_documents=relevant_docs, question=user_query)
    return response.strip() or "I don't know"
