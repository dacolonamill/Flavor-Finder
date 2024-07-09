
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import logging
import os
import pdb

app = Flask(__name__)

if torch.cuda.is_available():
    MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda")
    device = torch.device("cuda")
    print('Cuda is Available. Model is:', MODEL_NAME)
else:
    print("CUDA is not available. Exiting.")
    exit(1)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

model_name = 'Alibaba-NLP/gte-large-en-v1.5'
embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'trust_remote_code': True, 'device': device})

def chroma_search(query, chroma_dir):
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
    docs = db.similarity_search(query, k=5)
    context = ''
    print('\nThe most relevant reviews are:\n')
    for i, doc in enumerate(docs):
        print('Review #', i + 1, doc.page_content)
        context += 'Reviewer ' + str(i + 1) + ' says: ' + doc.page_content + '. '
    print(context)
    return context

def get_Request_response(text):
    query = text
    db_select = 'Y'
    search_db = chroma_search

    if db_select[0] == 'Y':
        chroma_dir = '/database/PA_200c_named_db'
        db_dir = chroma_dir
        try:
            context = search_db(query, db_dir)
        except Exception as e:
            app.logger.error(f"Error in search_db: {e}")
            context = ''
    else:
        context = ''

    extra_instruction = ". (Answer in just two sentences, maximum. Avoid answering if the question is not food-related. And please summarize the following reviews to guide your answer:)"
    template = """<s>[INST] <<SYS>>""" + query + extra_instruction + """<</SYS>>{text} [/INST]"""

    try:
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template,
        )
        app.logger.debug(f"Context: {context}")
        app.logger.debug(f"Prompt: {prompt.format(text=context)}")
        out = llm.invoke(prompt.format(text=context))
        app.logger.debug(f"Output: {out}")
        return '\n' + out.split('[/INST]  ')[1]
    except Exception as e:
        app.logger.error(f"Error in LLM invocation: {e}")
        return "Internal Server Error"

@app.route('/health', methods=['GET'])
def health_check():
    return 'Healthy', 200

@app.route("/")
def index_1st_stp():
    return render_template('index.html')

@app.route("/get_request", methods=["GET", "POST"])
def request_agin():
    input = request.form["msg"]
    return get_Request_response(input)

@app.route("/get_restaurant", methods=["GET", "POST"])
def request_chat():
    input = request.form["msg"]
    return get_Request_response(input)

@app.route("/reset", methods=["POST"])
def reset():
    return jsonify(success=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=8123)
