import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys,yaml,Utilities as ut
from PyPDF2 import PdfReader 

from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
from llama_index.core import SimpleDirectoryReader

documentvalue = ''
def load_pdf(docvalue):
   
   # Load the pdf file and split it into smaller chunks
   initdict={}
   initdict = ut.get_tokens()
   hf_token = initdict["hf_token"]
   embedding_model_id = initdict["embedding_model"]
   chromadbpath = initdict["chatPDF_chroma_db"]
   
   embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
   
   text_splitter = CharacterTextSplitter(separator="\n",chunk_size=700, chunk_overlap=70,length_function=len)
   chunks = text_splitter.split_text(docvalue)
   print("chunks length",chunks )
   # store it in chroma db
   db = Chroma.from_texts(chunks, embeddings, persist_directory=chromadbpath,collection_name="pdfstore")
   
   
   return db

# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.title("PatentGuru - PDF Ingestion")

# Access the uploaded ref via a key.
pdf_file = st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')
documentvalue=""
if ss.pdf:
    ss.pdf_ref = ss.pdf  # backup
    print("ss.pdf_ref  >>>",ss.pdf_ref)
    #print (os.path.join(".\tempDir",ss.pdf_ref.name))
   
    with open((os.path.join("./tempDir",ss.pdf_ref.name)),"wb") as f: 
        print("file name",f.name)
        f.write(ss.pdf.getbuffer())
   
    with open((os.path.join("./tempDir",ss.pdf_ref.name)),"rb") as file: 
        print("Inside file reader>>>",file.name)
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            
            documentvalue +=page.extract_text()
                 
    st.success("Saved File")
print("documentvalue  ",documentvalue)



# Now you can access "pdf_ref" anywhere in your app.
if ss.pdf_ref:
    binary_data = ss.pdf_ref.getvalue()
    pdf_viewer(input=binary_data, width=700,height=400)
    
# Main chat form
with st.form("chat_form"):
    submit_button = st.form_submit_button("Ingest this document")
    if submit_button:
        print(ss.pdf_ref.name)
        load_pdf(documentvalue)
        
        st.write ("Document Ingestion completed successfully")
        
