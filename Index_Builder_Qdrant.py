import os
import json
import hashlib
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import VectorParams, Distance, PointStruct, FilterSelector, MatchValue, FieldCondition, Filter

# ---- Constants ----
HASH_FILE = "data_hash.json"

embedding_model= SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

client= QdrantClient(host="localhost", port=6333)

#this function computes the hash for data files unless its system generated files and returns key-value pair of filepaths and theirs hashes
def compute_datafile_hashes(data_dir):
    file_hashes = {}
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.startswith(".") or file.endswith((".tmp", ".bak")) or "checkpoint" in file.lower():
                continue
            else:
                filepath = os.path.join(root, file)
                hash_md5 = hashlib.md5()          
                with open(filepath, "rb") as f:
                    while chunk := f.read(4096):   
                        hash_md5.update(chunk)
                file_hashes[filepath] = hash_md5.hexdigest()  
    return file_hashes

#this functions detects hash for data sources and return the full filepath of changed data soruces
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

#this function saves the hashes dictionary
def save_file_hashes(hash_file, new_hashes):
    with open(hash_file, "w") as f:
        json.dump(new_hashes, f)

#this function generate unique hashes combining file and row index for unique ids for each point in Qdrant DB
def generate_point_id(filepath, row_index):
    unique_string= f"{filepath}_{row_index}"
    hash_value= hashlib.md5(unique_string.encode()).hexdigest()
    return int(hash_value[:8], 16)

#this function removes the records of purged data from Qdrant DB
def deleted_files_index_update(deleted_files):
    for file in deleted_files:
        if "product" in file:
            client.delete(
            collection_name="product_catalog",
            points_selector= FilterSelector(
                filter= Filter(
                    must= [
                        FieldCondition(
                            key="source_file",
                            match= MatchValue(value= file)
                            )
                        ]
                    )
                )
            )
        elif "knowledge" in file:
            client.delete(
            collection_name="knowledgebase",
            points_selector= FilterSelector(
                filter= Filter(
                    must= [
                        FieldCondition(
                            key="source_file",
                            match= MatchValue(value= file)
                            )
                        ]
                    )
                )
            )    

    print("Records have been updated!")

#this function converts the CSV file data into points to be updated into the Qdrant DB
def reindex_csv(filepath):
    data= pd.read_csv(filepath)

    points= []
    for i, row in data.iterrows():
        text_to_embed= f"{row['Product_Name']} {row['Product_Description']}"
        vector= embedding_model.encode(text_to_embed).tolist()

        try:
            price= float(row["Product_Price_Cleaned"])
        except:
            price= 0.0

        point= PointStruct(
            id=generate_point_id(filepath, i),
            vector=vector,
            payload={
                "source_file": filepath,
                "product_price_float": price,
                **row.to_dict()
            }
        )
        points.append(point)
    return points

#this function splits the TEXT file data into chunks, embeds them before converting them into points for Qdrant DB
def reindex_txt(filepath):
    documents= []
    loader= TextLoader(file_path= filepath, encoding="utf-8")
    documents.extend(loader.load())

    print("Splitting textual information into chunks...")
    splitter= RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    split_docs= splitter.split_documents(documents)

    points= []

    for i, doc in enumerate(split_docs):
        vector= embedding_model.encode(doc.page_content).tolist()

        point= PointStruct(
            id= generate_point_id(filepath, i),
            vector= vector,
            payload= {
                "source_file": filepath
            }
        )
        points.append(point)
    return points

#this function updates all the new information converted into points to their respective Qdrant collections
def reindex_files(files_to_reindex):
    for filepath in files_to_reindex:
        if "products" in filepath:
            if filepath.endswith(".csv"):
                points_csv= reindex_csv(filepath)
                client.upsert(collection_name="product_catalog", points= points_csv)
            elif filepath.endswith(".txt"):
                points_txt= reindex_txt(filepath)
                client.upsert(collection_name="knowledgebase", points= points_txt)
        elif "knowledge" in filepath:
            if filepath.endswith(".csv"):
                points_csv= reindex_csv(filepath)
                client.upsert(collection_name="knowledgebase", points= points_csv)
            elif filepath.endswith(".txt"):
                points_txt= reindex_txt(filepath)
                client.upsert(collection_name="knowledgebase", points= points_txt)
    
    print("Reindexing complete!")

# This function brings everything together.
# Assumes:
# - Product data is in CSV files inside the product folder and Stored in product-catalog collection
# - Text files are processed and stored in knowledgebase collection
# These assumptions may change as new data sources are added.
def smart_index_loader(data_dir):
    new_files, deleted_files, modified_files, new_hashes= detect_changes(data_dir, HASH_FILE)

    if not client.collection_exists(collection_name="product_catalog") and not client.collection_exists(collection_name="knowledgebase"):
        client.create_collection(
            collection_name="product_catalog",
            vectors_config= VectorParams(
                size= 768,
                distance= Distance.COSINE
            )
        )

        client.create_collection(
            collection_name="knowledgebase",
            vectors_config= VectorParams(
                size= 768,
                distance= Distance.COSINE
            )
        )

        for root, _, files in os.walk(data_dir):
            for file in sorted(files):
                if file.startswith(".") or file.endswith((".tmp", ".bak")) or "checkpoint" in file.lower():
                    continue
                else:
                    filepath = os.path.join(root, file)
                    if "product" in filepath and filepath.endswith(".csv"):
                        product_points= reindex_csv(filepath)
                        client.upsert(collection_name="product_catalog", points=product_points)
                    elif "knowledge" in filepath and filepath.endswith(".csv"):
                        knowledge_points= reindex_csv(filepath)
                        client.upsert(collection_name="knowledgebase", points=knowledge_points)
                    else:
                        knowledge_points= reindex_txt(filepath)
                        client.upsert(collection_name="knowledgebase", points=knowledge_points)

        save_file_hashes(HASH_FILE, new_hashes)

    elif client.collection_exists(collection_name="product_catalog") and client.collection_exists(collection_name="knowledgebase"):
        if bool(deleted_files):
            print("Looks like some of the data has been purged!")
            deleted_files_index_update(deleted_files)
        elif (bool(new_files) or bool(modified_files)):
            files_to_reindex= list(new_files) + modified_files
            reindex_files(files_to_reindex)
        else:
            print("Looks like all the data is intact for consumption!")

        save_file_hashes(HASH_FILE, new_hashes)

    elif client.collection_exists(collection_name="product_catalog") and not client.collection_exists(collection_name="knowledgebase"):
        client.create_collection(
            collection_name="knowledgebase",
            vectors_config= VectorParams(
                size= 768,
                distance= Distance.COSINE
            )
        )

        if bool(deleted_files):
            print("Looks like some of the data has been purged!")
            deleted_files_index_update(deleted_files)
        elif (bool(new_files) or bool(modified_files)):
            files_to_reindex= list(new_files) + modified_files
            reindex_files(files_to_reindex)
        else:
            print("Looks like all the data is intact for consumption!")

        save_file_hashes(HASH_FILE, new_hashes)

    elif not client.collection_exists(collection_name="product_catalog") and client.collection_exists(collection_name="knowledgebase"):
        client.create_collection(
            collection_name="product_catalog",
            vectors_config= VectorParams(
                size= 768,
                distance= Distance.COSINE
            )
        )

        if bool(deleted_files):
            print("Looks like some of the data has been purged!")
            deleted_files_index_update(deleted_files)
        elif (bool(new_files) or bool(modified_files)):
            files_to_reindex= list(new_files) + modified_files
            reindex_files(files_to_reindex)
        else:
            print("Looks like all the data is intact for consumption!")

        save_file_hashes(HASH_FILE, new_hashes)


def main():
    DATA_DIR = os.path.abspath("Data")
    smart_index_loader(DATA_DIR)

if __name__ == "__main__":
    main()