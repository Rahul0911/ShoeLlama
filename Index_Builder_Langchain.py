import os
import json
import shutil
import hashlib
import pandas as pd
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# ---- Constants ----
HASH_FILE = "data_hash.json"

# ---- Embeddings ----
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ---- Hashing Utilities ----
def compute_data_hash(data_dir):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.startswith(".") or file.endswith((".tmp", ".bak")) or "checkpoint" in file.lower():
                continue
            path = os.path.join(root, file)
            hash_md5.update(file.encode())
            with open(path, 'rb') as f:
                while chunk := f.read(4096):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

def detect_data_change(data_dir, hash_file):
    current_hash = compute_data_hash(data_dir)
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            saved_hash = json.load(f).get("hash")
        return current_hash != saved_hash, current_hash
    else:
        return True, current_hash

def save_data_hash(hash_file, hash_value):
    with open(hash_file, 'w') as f:
        json.dump({"hash": hash_value}, f)

# ---- Index Creation ----
def load_csv_as_documents(filepath):
    df = pd.read_csv(filepath, encoding="utf-8")
    documents = []

    for _, row in df.iterrows():
        # Construct a detailed string for each shoe
        content = (
            f"Name: {row.get('Product_Name', 'N/A')}\n"
            f"Price: ${row.get('Product_Price', 'N/A')}\n"
            f"Description: {row.get('Product_Description', 'No description available')}\n"
            f"Link: {row.get('Product_Link', 'No link available')}"
        )
        doc = Document(page_content=content, metadata={"source": filepath})
        documents.append(doc)

    return documents


def create_langchain_index(data_dir, persist_dir):
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
    splitter= RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    split_docs= splitter.split_documents(documents)

    print("Building a new FAISS Index...")
    vectordb= FAISS.from_documents(split_docs, embedding=embed_model)

    print("Saving FAISS index locally...")
    vectordb.save_local(persist_dir)
    return vectordb

def load_langchain_index(persist_dir):
    return FAISS.load_local(folder_path=persist_dir, embeddings=embed_model, allow_dangerous_deserialization=True)

# ---- Smart Loader ----
def smart_index_loader(data_dir, vector_store_dir):
    hash_changed, new_hash = detect_data_change(data_dir, HASH_FILE)

    # Define expected files
    index_faiss = os.path.join(vector_store_dir, "index.faiss")
    index_pkl = os.path.join(vector_store_dir, "index.pkl")

    index_files_exist = os.path.exists(index_faiss) and os.path.exists(index_pkl)

    if not index_files_exist:
        print("Index files missing! Rebuilding index from scratch...")
        if os.path.exists(vector_store_dir):
            shutil.rmtree(vector_store_dir)  # Clear any partial data
        os.makedirs(vector_store_dir, exist_ok=True)
        index = create_langchain_index(data_dir, vector_store_dir)
        save_data_hash(HASH_FILE, new_hash)

    elif hash_changed:
        print("Data change detected! Rebuilding index...")
        shutil.rmtree(vector_store_dir)
        os.makedirs(vector_store_dir, exist_ok=True)
        index = create_langchain_index(data_dir, vector_store_dir)
        save_data_hash(HASH_FILE, new_hash)

    else:
        print("No Data Change Detected -- Loading Existing Vector Index...")
        index = load_langchain_index(vector_store_dir)

    return index


# ---- Main Entrypoint ----
def main():
    DATA_DIR = os.path.abspath("Data")
    VECTOR_STORE_DIR = os.path.abspath("faiss_index")

    index = smart_index_loader(DATA_DIR, VECTOR_STORE_DIR)

if __name__ == "__main__":
    main()
