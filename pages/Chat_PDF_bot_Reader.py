# import required libraries
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
#from langchain.text_splitter import NLTKTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import Utilities as ut

from groq import Groq

import streamlit as st
import sys,yaml,json
import chromadb

initdict={}
initdict = ut.get_tokens()

groq_api_key = initdict["groq_api"]
chatmodel = initdict["chat_model"]

messages = [
    {"role": "system", "content": "You are an assistant designed to provide answers from the provided context. If the answer is not available in the context, do not make up an answer"}
]
client = Groq(
    api_key=groq_api_key,
)


def filterdistance(distcoll):
    myemptydict={}
    if len(distcoll) < 0:myemptydict
    for distances in distcoll['distances']:
        for distance in distances:
            if distance<50: return distcoll
            else: return myemptydict
           

def get_data(query):
    chat_history = []
    initdict={}
    initdict = ut.get_tokens()
    hf_token = initdict["hf_token"]
    embedding_model_id = initdict["embedding_model"]
    chromadbpath = initdict["chatPDF_chroma_db"]
    llm_repo_id = initdict["pdf_chat_model"]
    groq_api_key = initdict["groq_api"]
    embedding_model_id = initdict["embedding_model"]
    # We will use HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

    embedding_model = SentenceTransformer(embedding_model_id)
    chroma_client = chromadb.PersistentClient(path = chromadbpath)
    collection = chroma_client.get_collection(name = "pdfstore")
 
    #collection = chroma_client.get_or_create_collection(name=chromadbcollname)
    query_vector = embedding_model.encode(query).tolist()
    output = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        #where={"distances": "is_less_than_1"},
        include=['documents','distances'],
        
        )
    print (output)
    #Filter for distances
    #output = filterdistance(output)
    return json.dumps(output)
    
        
st.title("PatentGuru PDF Reader")

if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun   
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        #print("history Messages:::",message["content"])
        st.markdown(message["content"])
# Main chat form

#query = st.chat_input()
clear_history = st.checkbox('Clear previous chat history') 
if clear_history:
        st.session_state.messages = []
        st.write("Cleared previous chat history")
if query := st.chat_input():
   
    
    st.chat_message("user").write(query)
    response = get_data(query)
    print(response)
    
    if len(response)>0:
        prompt= 'Answer the question'+ query + 'only if its available in the provided context. Do not provide answer if it is not available in the provided context. The provided context is:' + response
        messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "user", "content": query})
        chat_completion = client.chat.completions.create(
        messages=messages,
        model=chatmodel,
        )
        response = chat_completion.choices[0].message.content
        st.write(response)
        # Add the response to the messages as an Assistant Role
        messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    else: 
        # write results
        st.write ("No results")


