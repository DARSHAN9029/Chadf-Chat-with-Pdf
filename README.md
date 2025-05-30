# 🗂️ Chat with PDF (with History)

A Conversational RAG (Retrieval-Augmented Generation) Streamlit app that allows users to upload PDFs and chat with them. The app maintains per-session **chat history** using LangChain and supports **context-aware queries** using Groq's `Gemma2-9b-It` LLM.

---

## 🚀 Features

- 📄 Upload and chat with one or more PDF documents.
- 💬 Maintains chat history across sessions (`ChatMessageHistory`).
- 🔁 History-aware query reformulation.
- 🤖 Powered by Groq's Gemma2-9b-It model.
- 🧠 Uses HuggingFace embeddings with Chroma vector store.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/chat-with-pdf-history.git
cd chat-with-pdf-history
pip install -r requirements.txt
```
---
🔐 Environment Variables
Create a .env file in the root directory:

.env:
HF_TOKEN=your_huggingface_token

---

▶️ How to Run
```bash
streamlit run app.py
```
Enter your Groq API key.
Upload one or more PDF files.
Provide a session_id to track history.
Ask questions based on the content.
Review full chat history below the answer.

---

✨ Credits

LangChain
Groq
Chroma
HuggingFace
Streamlit
