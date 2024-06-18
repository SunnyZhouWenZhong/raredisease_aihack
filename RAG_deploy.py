# 我们在`BCEmbedding`中提供langchain直接集成的接口。
from BCEmbedding.tools.langchain import BCERerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import ModelScopeEmbeddings
import os

from fastapi import FastAPI, Body
import uvicorn

app = FastAPI()

@app.post("/rag")
def rerank_gen(query=Body(None)):
    if "Hypophosphatasia" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database_chunk400"
    elif "hypophosphatasia" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database_chunk400"
    elif "HPP" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database_chunk400"
    elif "hpp" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database_chunk400"
    elif "Ehlers Danlos Syndrome" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database_chunk400"
    elif "EDS" in query['query']:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database_chunk400"
    else:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_EDS_database_chunk400"

    # init embedding model
    embedding_model_name = '/data/zhouwz/llm_model/RAG_model/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:5'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

    embed_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
    )

    faiss_db = FAISS.load_local(file_dir_path, embed_model)

    reranker_args = {'model': '/data/zhouwz/llm_model/RAG_model/bce-reranker-base_v1', 'top_n': 3, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)

    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )
    
    # response = compression_retriever.get_relevant_documents(query.decode()) # .decode()为post请求时的Body格式为text时使用
    response = compression_retriever.get_relevant_documents(query['query'])
    return response
if __name__ == '__main__':
    # uvicorn.run(app="RAG_deploy:app", port=8899)
    uvicorn.run(app="RAG_deploy:app", host="10.131.102.26", port=8899, reload=False)