import sys,yaml
import uuid
import Utilities as ut
import os
import streamlit as st

from groq import Groq

initdict={}
initdict = ut.get_tokens()
groq_api_key = initdict["groq_api"]
chatmodel = initdict["chat_model"]

messages = [
    {"role": "system", "content": "You are a patent advisor and expert"}
]
client = Groq(
    api_key=groq_api_key,
)

st.title("PatentGuru - Intelligent Chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # Add the user's question to the messages as a User Role
        messages.append({"role": "user", "content": prompt})
        chat_completion = client.chat.completions.create(
        messages=messages,
        model=chatmodel,
        )
        response = chat_completion.choices[0].message.content
        st.write(response)
        # Add the response to the messages as an Assistant Role
        messages.append({"role": "assistant", "content": response})