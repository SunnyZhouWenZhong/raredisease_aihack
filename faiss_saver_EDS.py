# 我们在`BCEmbedding`中提供langchain直接集成的接口。
from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PDFMinerLoader
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever

import os



# EDS_Papers_PDF_via_Liezl_Puzon PDF文档读取
file_dir_path2 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/EDS_Papers_PDF_via_Liezl_Puzon"
files2 = os.listdir(file_dir_path2)
print("length of files2: ", len(files2))
files2.sort()

for i in range(len(files2)):
    try:
        join_path2 = os.path.join(file_dir_path2, files2[i])
        print("the name of file: {}".format(files2[i]))

        documents = PDFMinerLoader(join_path2).load()

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
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_2_chunk600/2EDS_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_2_chunk600/2EDS_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("EDS_Papers_PDF_via_Liezl_Puzon PDF文档读取成功!")
# EDS_Papers_PDF_via_Liezl_Puzon 已经全部成功存入Faiss库，其中有小部分的PDF损坏就没有存了。


# Sam_Keating_EDS_and_Comorbidities_Dataset_PDF PDF文档读取
file_dir_path5 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/Sam_Keating_EDS_and_Comorbidities_Dataset_PDF"
files5 = os.listdir(file_dir_path5)
print("length of files5: ", len(files5))
files5.sort()

for i in range(len(files5)):
    try:
        join_path5 = os.path.join(file_dir_path5, files5[i])
        print("the name of file: {}".format(files5[i]))

        documents = PDFMinerLoader(join_path5).load()

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
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_5_chunk600/5Sam_Keating_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_5_chunk600/5Sam_Keating_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("Sam_Keating_EDS_and_Comorbidities_Dataset_PDF PDF文档读取成功!")


# Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_1 PDF文档读取
file_dir_path61 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_1"
files61 = os.listdir(file_dir_path61)
print("length of files61: ", len(files61))
files61.sort()

for i in range(len(files61)):
    try:
        join_path61 = os.path.join(file_dir_path61, files61[i])
        print("the name of file: {}".format(files61[i]))

        documents = PDFMinerLoader(join_path61).load()

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
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_1_chunk600/6EDS_1_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_1_chunk600/6EDS_1_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_1 PDF文档读取成功!")


# Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_2 PDF文档读取
file_dir_path62 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_2"
files62 = os.listdir(file_dir_path62)
print("length of files62: ", len(files62))
files62.sort()

for i in range(len(files62)):
    try:
        join_path62 = os.path.join(file_dir_path62, files62[i])
        print("the name of file: {}".format(files62[i]))

        documents = PDFMinerLoader(join_path62).load()

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
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_2_chunk600/6EDS_2_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_2_chunk600/6EDS_2_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_2 PDF文档读取成功!")


# Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_3 PDF文档读取
file_dir_path63 = "/data/zhouwz/hjbcompetition/dataset_hanjianbing/Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_3"
files63 = os.listdir(file_dir_path63)
print("length of files63: ", len(files63))
files63.sort()

for i in range(len(files63)):
    try:
        join_path63 = os.path.join(file_dir_path63, files63[i])
        print("the name of file: {}".format(files63[i]))

        documents = PDFMinerLoader(join_path63).load()

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
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

        embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                            model_kwargs=embedding_model_kwargs,
                                            encode_kwargs=embedding_encode_kwargs)
        
        vectordb = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectordb.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_3_chunk600/6EDS_3_chunk600_{}".format(i+1))
        print("/data/zhouwz/hjbcompetition/sourcecode/faiss_database_chunk600/EDS_6_3_chunk600/6EDS_3_chunk600_{}".format(i+1))
    except:
        pass
    
    continue

print("Sam_Keating_EDS_Paper_Archive_PDF/Ehlers_Danlos_3 PDF文档读取成功!")