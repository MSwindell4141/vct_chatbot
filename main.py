'''
Project: VCT Chatbot
Author: Micah Swindell
Date: 2024-09-16
Credit:
    - https://www.youtube.com/watch?v=E1-mUfpeRu0
    - https://github.com/trevorspires/Bedrock-Chatbot-Youtube

Description: Start of my project for the VCT Hackathon.
First attempt at creating and learning about chatbots.
This is a simple chatbot that uses the Bedrock API to generate responses to user input.
The chatbot is built using the Langchain library which is a wrapper around the Bedrock API. The chatbot is built using the Streamlit library for the web interface.
'''


from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st
import json


def main():
    config = load_config('config.json')
    os.environ["AWS_PROFILE"] = config.get("aws_profile")

    # bedrock client
    bedrock_client = boto3.client(
        service_name = "bedrock-runtime",
        region_name = "us-east-1"
    )
    modelID = "anthropic.claude-v2:1"
    llm = Bedrock(
        model_id = modelID,
        client = bedrock_client,
        model_kwargs = {"temperature": 0.5} ## arguemnts I can give to the model
    )

    run_web_app()

def load_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables = ["language", "freeform_text"],
        template = "In {language}, {freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})
    return response

def run_web_app():
    st.title("VCT Chatbot")
    language = st.sidebar.selectbox("Language", ["english", "french", "german", "spanish"])

    if language:
        freeform_text = st.sidebar.text_area(label="Enter your text here", max_chars=100)

    if freeform_text:
        response = my_chatbot(language, freeform_text)
        st.write(response["text"])