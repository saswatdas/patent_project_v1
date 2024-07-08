# import required libraries
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
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
import sys,yaml

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference 
        # you don't need it 
        #self.text+=token+"/" 
        self.text+=token
        self.container.markdown(self.text) 

def get_data(query):
    chat_history = []
    initdict={}
    initdict = ut.get_tokens()
    hf_token = initdict["hf_token"]
    embedding_model_id = initdict["embedding_model"]
    chromadbpath = initdict["chatPDF_chroma_db"]
    llm_repo_id = initdict["pdf_chat_model"]
    groq_api_key = initdict["groq_api"]
    
    # We will use HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

    #retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 1})
    # load from disk
    print(chromadbpath)
    db = Chroma(persist_directory=chromadbpath, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2})
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    chat_box=st.empty() 
    stream_handler = StreamHandler(chat_box)
    
    #llm = HuggingFaceHub(huggingfacehub_api_token=hf_token, 
    #                    repo_id=llm_repo_id,  callback_manager = [stream_handler], verbose=True, model_kwargs={"temperature":0.2, "max_new_tokens":256})

    
    #llm = Groq(api_key=groq_api_key, model="llama3-70b-8192")
    # Initialize a ChatGroq object with a temperature of 0 and the "mixtral-8x7b-32768" model.
    prompt = set_custom_prompt()

    chat_model = ChatGroq(temperature=0, model_name=llm_repo_id,api_key=groq_api_key)


    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                chain_type="stuff",
                                retriever=retriever,
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": prompt})

    response = qa.invoke({"query": query})
    return response
    
st.title("PatentGuru PDF Reader")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("Chat with PDF: ")
    clear_history = st.checkbox('Clear previous chat history') 
    submit_button = st.form_submit_button("Send")    

if submit_button:
    if clear_history:
        st.write("Cleared previous chat history")
    
    response = get_data(query)
    print(response)
    
    if len(response)>0:
        st.write(response['result'])
        if response['result'].upper()=="I DON'T KNOW.":
            st.write("The requested information is not available in my PDF source document repository")           
    else: 
        # write results
        st.write ("No results")


