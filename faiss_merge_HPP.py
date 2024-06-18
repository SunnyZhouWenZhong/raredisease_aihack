from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

file_dir_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_4_1"
files = os.listdir(file_dir_path)
files.sort()
print("files: \n", files)

# init embedding model
embedding_model_name = '/data/zhouwz/LLMmodel/RAG_model/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cuda:2'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

embed_model = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

join_path = "/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database/HPP_PDF" 
faiss_db = FAISS.load_local(join_path, embed_model)

for i in range(len(files)):
    faiss_path = os.path.join(file_dir_path, files[i]) 
    db = FAISS.load_local(faiss_path, embed_model)
    faiss_db.merge_from(db)
    # print("{}: \n\n".format(i), faiss_db.docstore._dict)
    faiss_db.save_local("/data/zhouwz/hjbcompetition/sourcecode/faiss_database/HPP_database")