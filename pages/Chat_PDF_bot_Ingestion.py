import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.llms import HuggingFaceHub
#from langchain_community.vectorstores  import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.chains import ConversationalRetrievalChain
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
import sys,yaml,Utilities as ut

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from llama_index.core import SimpleDirectoryReader


def load_pdf(filename):
   
   # Load the pdf file and split it into smaller chunks
   initdict={}
   initdict = ut.get_tokens()
   hf_token = initdict["hf_token"]
   embedding_model_id = initdict["embedding_model"]
   chromadbpath = initdict["chatPDF_chroma_db"]
   
   embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
   loader = DirectoryLoader('./tempDir/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)

   documents = loader.load()
   
   #print (len(documents))
   
   # Split the documents into smaller chunks 

   text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
   texts = text_splitter.split_documents(documents)
    
   #Using Chroma vector database to store and retrieve embeddings of our text
   db = Chroma.from_documents(texts, embeddings, persist_directory=chromadbpath)
   return db

# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.title("PatentGuru - PDF Ingestion")

# Access the uploaded ref via a key.
pdf_file = st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

if ss.pdf:
    ss.pdf_ref = ss.pdf  # backup
    #print (os.path.join(".\tempDir",ss.pdf_ref.name))
    with open((os.path.join("./tempDir",ss.pdf_ref.name)),"wb") as f: 
        f.write(ss.pdf.getbuffer())
    st.success("Saved File")
    
# Now you can access "pdf_ref" anywhere in your app.
if ss.pdf_ref:
    binary_data = ss.pdf_ref.getvalue()
    pdf_viewer(input=binary_data, width=700,height=400)
    
# Main chat form
with st.form("chat_form"):
    submit_button = st.form_submit_button("Ingest this document")
    if submit_button:
        print(ss.pdf_ref.name)
        load_pdf(ss.pdf_ref.name)
        st.write ("Document Ingestion completed successfully")
