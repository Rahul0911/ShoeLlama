from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
import chromadb
import os 
import hashlib 
import json
import shutil

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
HASH_FILE = "data_hash.json"

def create_index(data_dir, vector_store_index, collection_name):
    
    documents = SimpleDirectoryReader(data_dir).load_data()

    chroma_client = chromadb.PersistentClient(path=vector_store_index)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store= ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(documents, embed_model=embed_model, storage_context=storage_context)

    return index


def load_existing_index(vector_store_dir, collection_name):
    chroma_client = chromadb.PersistentClient(path=vector_store_dir)
    load_collection= chroma_client.get_collection(name=collection_name)

    load_vector_store = ChromaVectorStore(chroma_collection=load_collection)
    load_storage_context= StorageContext.from_defaults(vector_store=load_vector_store)

    saved_index = VectorStoreIndex.from_vector_store(
        vector_store=load_vector_store, 
        storage_context=load_storage_context, 
        embed_model=embed_model
        )

    return saved_index


def compute_data_hash(data_dir):
    hash_md5= hashlib.md5()
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
            saved_hash= json.load(f).get("hash")
        return current_hash!= saved_hash, current_hash
    else:
        return True, current_hash
    

def save_data_hash(hash_file, hash_value):
    with open(hash_file, 'w') as f:
        json.dump({"hash": hash_value}, f) 


def smart_index_loader(data_dir, vector_store_dir, collection_name):
    hash_changed, new_hash= detect_data_change(data_dir, HASH_FILE)

    if hash_changed:
        print("New data detected! -- Rebuilding Vector Index from Scratch...")
        if os.path.exists(vector_store_dir):
            shutil.rmtree(vector_store_dir)
        index= create_index(data_dir, vector_store_dir, collection_name)
        save_data_hash(HASH_FILE, new_hash)
    else:
        print("No data change detected -- Loading existing data..")
        index=load_existing_index(vector_store_dir, collection_name)

    return index


def main():
    DATA_DIR = os.path.abspath("Data")
    VECTOR_STORE_DIR = os.path.abspath("./chromaDB")
    COLLECTION_NAME= "my_collection"

    index = smart_index_loader(DATA_DIR, VECTOR_STORE_DIR, COLLECTION_NAME)
    print("\n INDEX LOADED SUCCESSFULLY!")

if __name__ == "__main__":
    main()