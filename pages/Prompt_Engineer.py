from langchain.llms import HuggingFaceHub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
#from langchain_community.llms import LlamaCpp
#from langchain.chains import RetrievalQA
#from langchain_community.embeddings import SentenceTransformerEmbeddings

#from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

#from langchain.schema import HumanMessage

import os
import json,streamlit as st,Utilities as ut
from pathlib import Path

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

st.title("Prompt Engineer")

# Main chat form
with st.form("chat_form"):

    query = st.text_input("Enter the topic you want to generate prompt for?: ")
    #LLM_Summary = st.checkbox('Summarize results with LLM') 
    submit_button = st.form_submit_button("Send")    

     
    template = """
    <s>[INST] <<SYS>>
    Act as a patent advisor by providing subject matter expertise on any topic. Provide detailed and elaborate answers
    <</SYS>>

    {text} [/INST]
    """
    response=""
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    text = "Help me create a good prompt for the following: Information that is needed to file a US patent application for " + query
    #print(prompt.format(text=query))

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    initdict={}
    initdict = ut.get_tokens()
    #local_model_path = initdict["local_model_path"]
    hf_token = initdict["hf_token"]
    llm_repo_id = initdict["llm_repoid"]
    chat_box=st.empty() 
    stream_handler = StreamHandler(chat_box)
    
    # llm = LlamaCpp(
    #     model_path=local_model_path,
    #     temperature=0.8,
    #     max_tokens=500,
    #     top_p=1,
    #     #streaming=True,
    #     #callback_manager=callback_manager,
    #     callback_manager = [stream_handler],
    #     verbose=True,  # Verbose is required to pass to the callback manager
    # )
    llm = HuggingFaceHub(huggingfacehub_api_token=hf_token,repo_id=llm_repo_id,callback_manager = [stream_handler],verbose=True,model_kwargs={"temperature":1, "max_new_tokens":500,"streaming":True})
    
if submit_button:
    output = llm.invoke(prompt.format(text=text))
    print(output)
    if len(output)>0:
        substring = (output.lower()).partition("prompt:")[-1]
        if len(substring)>0: 
            response = "Potential Prompt:   \n\n\n" + str(substring)
        else: 
            response = "No results"
    else:
        response = "No results"
    
    st.write(response)



    
    
