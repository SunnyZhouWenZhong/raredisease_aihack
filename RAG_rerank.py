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

##打印文本内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

def rag_rerank(query):
    if "Hypophosphatasia" in query:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database_chunk400"
    elif "Ehlers Danlos Syndrome" in query:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database_chunk400"
    else:
        file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_EDS_database_chunk400"

    # init embedding model
    embedding_model_name = '/data/zhouwz/llm_model/RAG_model/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:6'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

    embed_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
    )

    faiss_db = FAISS.load_local(file_dir_path, embed_model)

    reranker_args = {'model': '/data/zhouwz/llm_model/RAG_model/bce-reranker-base_v1', 'top_n': 3, 'device': 'cuda:6'}
    reranker = BCERerank(**reranker_args)

    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )
    
    # response = compression_retriever.get_relevant_documents(query.decode()) # .decode()为post请求时的Body格式为text时使用
    response = compression_retriever.get_relevant_documents(query)
    return response
if __name__ == '__main__':
    question = "Will Ehlers Danlos syndrome lead to cardiovascular problems?"
    response = rag_rerank(question)

    print("response: \n", response)
    print("response: \n", pretty_print_docs(response))
    print("response1: \n", response[0].metadata['relevance_score'])