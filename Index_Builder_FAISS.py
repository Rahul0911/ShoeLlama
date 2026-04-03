import os
import json
import shutil
import hashlib
import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ---- Constants ----
HASH_FILE = "data_hash.json"

# ---- Embeddings ----
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def compute_datafile_hashes(data_dir):
    file_hashes = {}
    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if file.startswith(".") or file.endswith((".tmp", ".bak")) or "checkpoint" in file.lower():
                continue
            else:
                filepath = os.path.join(root, file)
                hash_md5 = hashlib.md5()          # fresh blender per file
                with open(filepath, "rb") as f:
                    while chunk := f.read(4096):   # memory efficient
                        hash_md5.update(chunk)
                file_hashes[filepath] = hash_md5.hexdigest()  # fingerprint string
    return file_hashes

def detect_changes(data_dir, hash_file):
    new_hashes = compute_datafile_hashes(data_dir)

    if not os.path.exists(hash_file):
        return set(new_hashes.keys()), set(), set(), new_hashes# first run
    else:
        with open(hash_file, "r") as f:
            old_hashes = json.load(f)
        
        old_keys = set(old_hashes.keys())
        new_keys = set(new_hashes.keys())
        
        new_files_added = new_keys - old_keys
        deleted_files = old_keys - new_keys
        modified_files = []
        for file in old_keys.intersection(new_keys):
            if old_hashes[file] != new_hashes[file]:
                modified_files.append(file)

        return new_files_added, deleted_files, modified_files, new_hashes
    
def save_file_hashes(hash_file, new_hashes):
    with open(hash_file, "w") as f:
        json.dump(new_hashes, f)

def load_csv_as_documents(filepath):
    documents= []
    data = pd.read_csv(filepath, encoding="utf-8")

    for _, rows in data.iterrows():
        content= "|".join(str(val) for val in rows.values)
        docs= Document(page_content=content, metadata= {"source": filepath})
        documents.append(docs)

    return documents

def reindex_files(files_to_reindex, existing_index, vector_store_dir):
    documents=[]
    for filepath in files_to_reindex:
        if filepath.endswith(".csv"):
            documents.extend(load_csv_as_documents(filepath))
        elif filepath.endswith(".txt"):
            loader= TextLoader(file_path=filepath, encoding="utf-8")
            documents.extend(loader.load())

    print("Splitting Modified Documents...")
    splitter= RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    split_docs= splitter.split_documents(documents=documents)

    print("Building Temporary FAISS index...")
    tempDB= FAISS.from_documents(split_docs, embedding=embed_model)

    print("Merging with the existing Index...")
    existing_index.merge_from(tempDB)

    print("Saving FAISS index locally...")
    existing_index.save_local(vector_store_dir)
    return existing_index

def create_langchain_index(data_dir, vector_store_dir):
    print("Loading Documents...")
    documents = []
    for filename in os.listdir(data_dir):
        filepath= os.path.join(data_dir, filename)
        if filename.endswith(".csv"):
            documents.extend(load_csv_as_documents(filepath))
        elif filename.endswith(".txt"):
            loader= TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    print("Splitting Documents...")
    splitter= RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    split_docs= splitter.split_documents(documents)

    print("Building a new FAISS Index...")
    vectordb= FAISS.from_documents(split_docs, embedding=embed_model)

    print("Saving FAISS index locally...")
    vectordb.save_local(vector_store_dir)
    return vectordb

def load_langchain_index(vector_store_dir):
    return FAISS.load_local(folder_path=vector_store_dir, embeddings=embed_model, allow_dangerous_deserialization=True)

def smart_index_loader(data_dir, vector_store_dir):
    index_faiss = os.path.join(vector_store_dir, "index.faiss")
    index_pkl = os.path.join(vector_store_dir, "index.pkl")

    index_files_exist = os.path.exists(index_faiss) and os.path.exists(index_pkl)

    new_files, deleted_files, modified_files, new_hashes= detect_changes(data_dir, HASH_FILE)

    if not index_files_exist:
        print("Oh no! It seems like Index files are missing. \n Rebuilding Index from Scratch...")
        if os.path.exists(vector_store_dir):
            shutil.rmtree(vector_store_dir)
        os.makedirs(vector_store_dir, exist_ok=True)
        index= create_langchain_index(data_dir, vector_store_dir)
        save_file_hashes(HASH_FILE, new_hashes)
    elif (bool(deleted_files)):
        print("Oops, Some files have been deleted. Rebuilding Index from scratch...")
        shutil.rmtree(vector_store_dir)
        os.makedirs(vector_store_dir, exist_ok=True)
        index= create_langchain_index(data_dir, vector_store_dir)
        save_file_hashes(HASH_FILE, new_hashes)
    elif (bool(new_files) or bool(modified_files)):
        print("Oops, Some files have been modified or add to the data source. \n Merging changes with the exisiting index...")
        files_to_reindex= new_files | modified_files
        existing_index= load_langchain_index(vector_store_dir)
        index= reindex_files(files_to_reindex, existing_index, data_dir)
        save_file_hashes(HASH_FILE, new_hashes)
    else:
        print('Yipee, No changes in data source! Loading existing Index')
        index= load_langchain_index(vector_store_dir)

    return index

# ----Main Entrypoint-----
def main():
    DATA_DIR= os.path.abspath("Data")
    VECTOR_STORE_DIR= os.path.abspath("faiss_index")

    index= smart_index_loader(DATA_DIR, VECTOR_STORE_DIR)

if __name__ == "__main__":
    main()