# 我们在`BCEmbedding`中提供langchain直接集成的接口。
from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PDFMinerLoader
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever

import os


# HPP_Papers_PDF_Via_Liezl_Puzon PDF文档读取
file_dir_path4 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/HPP_Datasets/HPP_Papers_PDF_Via_Liezl_Puzon"
files4 = os.listdir(file_dir_path4)
print("length of files4: ", len(files4))
files4.sort()

for i in range(len(files4)):
    try:
        join_path4 = os.path.join(file_dir_path4, files4[i])
        print("the name of file: {}".format(files4[i]))

        documents = PDFMinerLoader(join_path4).load()

        key_word0 = "ABSTRACT"
        key_word1 = "Abstract"
        key_word2 = "Reference"
        key_word3 = "REFERENCE"

        if key_word0 in documents[0].page_content:
             index1 = documents[0].page_content.find(key_word0)
        elif key_word1 in documents[0].page_content:
             index1 = documents[0].page_content.find(key_word1)
        else:
             index1 = 0
            
        if key_word2 in documents[0].page_content:
             index2 = documents[0].page_content.find(key_word2)
        elif key_word3 in documents[0].page_content:
            index2 = documents[0].page_content.find(key_word3)
        else:
            index2 = -1
        
        documents[0].page_content = documents[0].page_content[index1+len(key_word1)+1:index2]
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0, length_function = len, 
                                                            separators = [".", ",", " ", ""], keep_separator = True)
        texts = text_splitter.split_documents(documents)

            # init embedding model
        embedding_model_name = '/data/zhouwz/LLMmodel/RAG_model/bce-embedding-base_v1'
        embedding_model_kwargs = {'device': 'cuda:2'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/HPP_4_2_chunk600/4HPP_2_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/HPP_4_2_chunk600/4HPP_2_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("HPP_Papers_PDF_Via_Liezl_Puzon PDF文档读取成功!")


# HPP_Mega_Set_PubMed_Records_TXT_via_Jacob_Cole_and_RPC TXT文档读取
file_dir_path4_2 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/HPP_Datasets/HPP_Mega_Set_PubMed_Records_TXT_via_Jacob_Cole_and_RPC"
files4_2 = os.listdir(file_dir_path4_2)
print("length of files4_2: ", len(files4_2))
files4_2.sort()

for i in range(len(files4_2)):
    try:
        join_path4_2 = os.path.join(file_dir_path4_2, files4_2[i])
        print("the name of file: {}".format(files4_2[i]))

        documents = TextLoader(join_path4_2).load()

        key_word0 = "Introduction"
        key_word1 = "INTRODUCTION"
        key_word2 = "Body"
        key_word3 = "Reference"
        key_word4 = "REFERENCE"
        key_word5 = "Ref"

        if key_word0 in documents[0].page_content:
                index1 = documents[0].page_content.find(key_word0)
        elif key_word1 in documents[0].page_content:
                index1 = documents[0].page_content.find(key_word1)
        elif key_word2 in documents[0].page_content:
                index1 = documents[0].page_content.find(key_word2)
        else:
                index1 = 0
            
        if key_word3 in documents[0].page_content:
                index2 = documents[0].page_content.find(key_word3)
        elif key_word4 in documents[0].page_content:
                index2 = documents[0].page_content.find(key_word4)
        elif key_word5 in documents[0].page_content:
                index2 = documents[0].page_content.find(key_word5)
        else:
                index2 = -1

        documents[0].page_content = documents[0].page_content[index1+len(key_word1)+1:index2]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function = len, 
                                                            separators = [".", ",", " ", ""], keep_separator = True)
        texts = text_splitter.split_documents(documents)

            # init embedding model
        embedding_model_name = '/data/zhouwz/LLMmodel/RAG_model/bce-embedding-base_v1'
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)

        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_4_1/4HPP_1_chunk500_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_4_1/4HPP_1_chunk500_{}".format(i+1))
    except:
        pass
    
    continue

print("HPP_Mega_Set_PubMed_Records_TXT_via_Jacob_Cole_and_RPC TXT文档读取成功!")