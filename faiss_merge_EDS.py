from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

file_dir_path1 = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_2"
files1 = os.listdir(file_dir_path1)
files1.sort()
print("files1: \n", files1)

# init embedding model
embedding_model_name = '/data/zhouwz/LLMmodel/RAG_model/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cuda:2'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

embed_model = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

join_path = os.path.join(file_dir_path1, files1[0])
faiss_db = FAISS.load_local(join_path, embed_model)

for i in range(1, len(files1)):
    faiss_path = os.path.join(file_dir_path1, files1[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database")
  
print("EDS_2 loading success!")


file_dir_path2 = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_5"
files2 = os.listdir(file_dir_path2)
files2.sort()
print("files2: \n", files2)

for i in range(len(files2)):
    faiss_path = os.path.join(file_dir_path2, files2[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database")

print("EDS_5 loading success!")

file_dir_path3 = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_6_1"
files3= os.listdir(file_dir_path3)
files3.sort()
print("files3: \n", files3)

for i in range(len(files3)):
    faiss_path = os.path.join(file_dir_path3, files3[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database")

print("EDS_6_1 loading success!")

file_dir_path4 = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_6_2"
files4= os.listdir(file_dir_path4)
files4.sort()
print("files4: \n", files4)

for i in range(len(files4)):
    faiss_path = os.path.join(file_dir_path4, files4[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database")

print("EDS_6_2 loading success!")

file_dir_path5 = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_6_3"
files5= os.listdir(file_dir_path5)
files5.sort()
print("files5: \n", files5)

for i in range(len(files5)):
    faiss_path = os.path.join(file_dir_path5, files5[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/EDS_database")

print("EDS_6_3 loading success!")