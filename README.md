# 🗂️ Chat with PDF (with History)

A Conversational RAG (Retrieval-Augmented Generation) Streamlit app that allows users to upload PDFs and chat with them. The app maintains per-session **chat history** using LangChain and supports **context-aware queries** using Groq's `Gemma2-9b-It` LLM.

---

## 🚀 Features

- 📄 Upload and chat with one or more PDF documents.
- 💬 Maintains chat history across sessions (`ChatMessageHistory`).
- 🔁 History-aware query reformulation.
- 🤖 Powered by Groq's Gemma2-9b-It model.
- 🧠 Uses HuggingFace embeddings with Chromadb as vector store database.

---

## 📦 Installation

```bash
git clone https://github.com/DARSHAN9029/Chadf-Chat-with-Pdf.git
```
---
```bash
pip install -r requirements.txt
```

---

🔐 Environment Variables
Create a .env file in the root directory:
```
HF_TOKEN=your_huggingface_token
```
---

▶️ How to Run
```bash
streamlit run app.py
```

## Steps:-
1. Enter your Groq API key.

2. Upload one or more PDF files.

3. Provide a session_id to track history.

4. Ask questions based on the content.

5. Review full chat history below the answer.

---
📸 Screenshots

![Screenshot 2025-05-30 163250](https://github.com/user-attachments/assets/01dec3ed-5c52-4c90-8356-2aa851dfd8d4)
![Screenshot 2025-05-30 163303](https://github.com/user-attachments/assets/d8228517-627c-41b9-99c9-05363ed81221)
![Screenshot 2025-05-30 163317](https://github.com/user-attachments/assets/092e46db-1fe6-425f-9c58-d4fdbe662bed)
