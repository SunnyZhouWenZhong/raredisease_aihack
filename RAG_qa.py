from openai import OpenAI
import csv
from RAG_rerank import rag_rerank

openai_api_key = "EMPTY" # 随便设，只是为了通过接口参数校验
openai_api_base = "http://localhost:5090/v1"
# openai_api_base = "http://10.131.102.17:5095/v1"
threshold = 0.45 # 此阈值为 rerank 检索到的前 k 个文档片段的相似度的阈值

question = "How do people fare after receiving treatment for hypophophosphatasia?"
response = rag_rerank(question)

# system_content = "you are an experienced and well-informed expert, answer the questions according to the documents."
system_content = "As an expert in rare diseases, respond to the question based on the provided documents."
if response[0].metadata['relevance_score'] >= threshold:
    if response[1].metadata['relevance_score'] >= threshold:
        if response[2].metadata['relevance_score'] >= threshold:
            user_content = """
            documents 1: 
            {}

            documents 2: 
            {}
            
            documents 3: 
            {}

            question:
            {}
            """.format(response[0].page_content, response[1].page_content, response[2].page_content, question)
        else:
            user_content = """
            documents 1: 
            {}

            documents 2: 
            {}

            question:
            {}
            """.format(response[0].page_content, response[1].page_content, question)
    else:
        user_content = """
        documents: 
        {}

        question:
        {}
        """.format(response[0].page_content, question)
else:
    user_content = """
    question:
    {}
    """.format(question)

prompts = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
# prompts = system_content+"\n###  Human: "+ user_content+ "\n### Assistant:"

print("prompts: \n", prompts)

client = OpenAI(
api_key=openai_api_key,
base_url=openai_api_base,
)
# /data/zhouwz/hjbcompetition/lora_merge_model/merge_llama3_8b_lora_bs6_lr0.0001_epoch20
# /data/zhouwz/llm_model/Meta-Llama-3-8B
chat_outputs = client.chat.completions.create(
    model="/data/zhouwz/hjbcompetition/lora_merge_model/llama3_8b_ai4rd",
    frequency_penalty = 1.0,
    max_tokens = 2048,
    presence_penalty = 1.0,
    temperature = 0.2,
    top_p = 0.5,
    stop=["\n\n", "#", "\n##", "\n4"],
    messages=prompts
)
print("output of the RAG+LLM: \n", chat_outputs.choices[0].message.content)