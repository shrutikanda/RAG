import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def load_documents():
    print("Loading data...")
    loader = DirectoryLoader(str(ROOT / "docs"), glob="*.txt", loader_cls=TextLoader,  loader_kwargs={"encoding": "utf-8"})

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    for i, doc in enumerate(documents[:3]):
        print(f"Document {i}: {doc.metadata['source']} - {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents



def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting documents...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    split_docs = text_splitter.split_documents(documents)

    print(f"Split into {len(split_docs)} chunks.")

    if split_docs:
        for i, doc in enumerate(split_docs[:3]):
            print(f"Chunk {i}: {doc.metadata['source']} - {doc.page_content[:100]}...")
            print(f"metadata: {doc.metadata}")
            print("...")

    return split_docs


def create_embeddings(chunks):
    print("Creating and persist embeddings in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Creating ChromaDB collection...")
    vector_store = Chroma.from_documents(collection_name="documents", 
                          documents=chunks,
                          embedding=embedding_model, 
                          persist_directory=str(ROOT / "db/chroma_db"),
                          collection_metadata={"hnsw:space": "cosine"})

    print(f"Vector store collection created with {vector_store._collection.count()} vectors.")

    return vector_store

if __name__ == "__main__":
   
   #Load documents
   documents =  load_documents()

   #Chunk documents
   chunks = split_documents(documents)

   #Create embeddings and persist in ChromaDB           
   vector_store = create_embeddings(chunks)