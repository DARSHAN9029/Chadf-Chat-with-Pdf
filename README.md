# 🗂️ CHAdf-Chat with PDF with chat history

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

## ▶️ How to Run
```bash
streamlit run app.py
```
Enter your Groq API key.
Upload one or more PDF files.
Provide a session_id to track history.
Ask questions based on the content.
Review full chat history below the answer.

---

## Screenshots
![Screenshot 2025-05-30 163250](https://github.com/user-attachments/assets/430d9755-c470-4bbe-a6b0-cf7ffc2007a5)
![Screenshot 2025-05-30 163303](https://github.com/user-attachments/assets/cc5df030-14ab-4608-95fe-976cf3ac3616)
![Screenshot 2025-05-30 163317](https://github.com/user-attachments/assets/f9d8e4a3-6924-49e3-8b3f-88bc1118b3d8)

---

## ✨ Credits

LangChain
Groq
Chroma
HuggingFace
Streamlit
