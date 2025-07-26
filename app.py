import os
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load and embed messages
df = pd.read_csv("message_reply_pairs.csv")
documents = [
    Document(page_content=f"User: {row['incoming']}\nBot: {row['your_reply']}")
    for _, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Streamlit UI
st.set_page_config(page_title="WhatsApp RAG Bot", layout="centered")
st.title("📱 WhatsApp RAG Reply Bot")
st.markdown("Ask anything like how you chat, and the bot will respond just like you.")

user_input = st.text_input("Your Message:", placeholder="Hey, what's up?", key="input")

if user_input:
    with st.spinner("Thinking..."):
        try:
            docs = retriever.get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
You are a helpful WhatsApp assistant. Given the following past message-response pairs:

{context}

Based on this, respond to the following message from the user:

User: {user_input}
Bot:"""

            response = gemini_model.generate_content(prompt)
            st.success("Bot:")
            st.write(response.text)

        except Exception as e:
            st.error(f"Error: {e}")
