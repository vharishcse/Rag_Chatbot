# chat_ui.py

import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag_chatbot import build_faiss_vector_store  # Import the function

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY is not None, "❌ GOOGLE_API_KEY not found in .env"

# Build FAISS DB if missing (e.g., first deploy on Render)
if not Path("faiss_db/index.faiss").exists():
    print("⚠️ FAISS index not found. Rebuilding...")
    build_faiss_vector_store()

# Load FAISS DB
vector_db = FAISS.load_local(
    "faiss_db",
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    allow_dangerous_deserialization=True
)

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY
)

# Chatbot function
def chatbot(user_input, history):
    if not user_input.strip():
        return "", history

    docs = vector_db.similarity_search(user_input, k=4)

    if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
        answer = "I don't know"
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are a strict support assistant for AngelOne. Use ONLY the context below to answer the question. If the answer is not explicitly present or relevant in the context, just respond with: "I don't know". 
        Context:
            {context}

        Question: {user_input}

        Answer:"""

        response = llm.invoke(prompt)
        answer = response.content.strip() or "I don't know"

    history.append((user_input, answer))
    return "", history

# Predefined example questions
examples = [
    "Why is my trading balance not updated even after adding funds successfully?",
    "Can I place an order when markets are closed (AMO Order)?",
    "What are brokerage charges?"
]

# Gradio UI
with gr.Blocks(title="📘 AngelOne Support Chatbot") as demo:
    gr.Markdown("## 💬 Chat with AngelOne Support Docs")

    chatbot_display = gr.Chatbot(label="AngelOne Docs Chat")
    user_input = gr.Textbox(placeholder="Ask a question about AngelOne support...", lines=2, show_label=False)
    send_btn = gr.Button("Send")
    state = gr.State([])

    # Example question buttons
    with gr.Row():
        for question in examples:
            gr.Button(question).click(
                lambda q, h: chatbot(q, h), 
                inputs=[gr.Textbox(value=question, visible=False), state], 
                outputs=[user_input, chatbot_display]
            )

    # Send btn and Enter key both submit
    send_btn.click(fn=chatbot, inputs=[user_input, state], outputs=[user_input, chatbot_display])
    user_input.submit(fn=chatbot, inputs=[user_input, state], outputs=[user_input, chatbot_display])

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
