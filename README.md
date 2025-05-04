# 📘 AngelOne Support Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built for AngelOne customer support. It uses LangChain, Gemini Flash, FAISS vector store, and Gradio to enable natural language Q&A over structured support FAQs.

---

## 🚀 Features

- **Google Gemini 1.5 Flash** used as the LLM.
- **FAISS Vector Store** to store and search document embeddings.
- **RAG-based** — retrieves relevant context before generating answers.
- **I don't know policy** — does not hallucinate, strictly follows support content.
- **Gradio UI** with:
  - Clickable example questions.
  - Chat interface with send button and Enter key submission.

---

## 🗂️ Project Structure

```
your-project/
├── .env                        # API key stored here
├── .gitignore                 # Ignore sensitive and generated files
├── angelone_faqs.csv          # CSV file containing support Q&A
├── faiss_db/                  # Vector database folder
├── venv/                      # Python virtual environment
├── requirements.txt           # All dependencies
├── rag_chatbot.py             # Script to generate vector DB from CSV
├── chat_ui.py                 # Gradio chatbot interface
├── README.md                  # You're reading it!
└── .gradio/                   # Gradio session files (ignored)
```

---

## ⚙️ Setup Instructions

1. **Clone this repo & enter directory**
```bash
git clone <your-repo-url>
cd your-project
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up `.env` file**
```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Prepare vector database**
```bash
python rag_chatbot.py
```

6. **Run the chatbot UI**
```bash
python chat_ui.py
```

Visit [http://127.0.0.1:7860](http://127.0.0.1:7860) to start chatting.

---

## 📄 Example Questions

- Why is my trading balance not updated even after adding funds successfully?
- How do I close my account?
- What are brokerage charges?

---

## 🧠 Tech Stack

- **LLM**: Google Gemini 1.5 Flash via `langchain-google-genai`
- **Embeddings**: `GoogleGenerativeAIEmbeddings`
- **RAG Engine**: LangChain + FAISS
- **Frontend**: Gradio Blocks-based chat interface

---

## ⚠️ Notes

- The chatbot answers **only** based on the uploaded FAQ CSV.
- Questions outside of this context return `"I don't know"` to avoid hallucinations.
* Make sure `.env`, `faiss_db/`, `venv/`, and `.gradio/` are listed in `.gitignore` to avoid pushing them to GitHub.

---

## 👨‍💻 **Author**  

**Harish Yadav** 🚀  
🔗 [LinkedIn](https://www.linkedin.com/in/v-harish-yadav-b2bb52241)  

---

## 📄 License

MIT License (or update if needed).