#importing libs
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

#embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#set up streamlit app
st.title("Conversational RAG and CHAT with Pdf with chat history")
st.write("Upload pdf's and chat with their content")

#Input the groq api key
api_key=st.text_input("Enter your Groq api key:",type="password")

#check if api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    #interface of the chat

    session_id=st.text_input("Session_id", value="default_session")

    #managing the chat history
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_file=st.file_uploader("Choose the PDF file",type="pdf",accept_multiple_files=True)

    #process uploaded files
    if uploaded_file:
        documents=[]

        for uploaded_file in uploaded_file:
            temp_pdf=f"./temp.pdf"
            
            with open(temp_pdf,"wb") as f:
                f.write(uploaded_file.getvalue())
                f_name=uploaded_file.name
            
            loader=PyPDFLoader(temp_pdf)        #pdf file loader
            docs=loader.load()
            documents.extend(docs)

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vector_store.as_retriever()


        contextualize_q_system_propmt=(

            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "                
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt= ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_propmt),
                MessagesPlaceholder("chat_history"),
                ('human',"{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        #Answer question

        system_promot=(

            "You are an assistant for quesyion-answering tasks."
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer , say that you "
            "don't know . Use three sentence maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt= ChatPromptTemplate.from_messages(

            [
                ("system",system_promot),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        #conversational rag chain
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        #create the user input
        user_input=st.text_input("Your question")

        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },

            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History",session_history.messages)
else:
    st.warning("Please enter the Groq Api Key")