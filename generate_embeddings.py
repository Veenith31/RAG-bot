from langchain_huggingface import HuggingFaceEmbeddings  # ✅ New import
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

df = pd.read_csv("message_reply_pairs.csv")

documents = [
    Document(page_content=row["incoming"], metadata={"reply": row["your_reply"]})
    for _, row in df.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding_model)
db.save_local("vector_store.faiss")
