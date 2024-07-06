from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import streamlit as st

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
   

class PDFVectoriser:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = None
        self.vector_db = None

    def extract_from_pdf(self, pdf_file):
        reader = PdfReader(pdf_file)
        texts = ""
        for page in reader.pages:
            texts += page.extract_text() or ""
        return texts

    def create_vector_db(self, texts):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', task_type='SEMANTIC_SIMILARITY', google_api_key=os.environ['GOOGLE_API_KEY'])
            v_db = FAISS.from_texts(texts, self.embeddings)
            return v_db
        except Exception as e:
            print(f"ERROR: Failed to create vector database: {e}")
            return None

    def get_similar_context(self, v_user, n=5, v_db = None):
        if v_user and v_db:
            docs = v_db.similarity_search(v_user, k=n)
            return docs
        return []
    
    def split_text(self, text):
        return self.text_splitter.split_text(text)
        
