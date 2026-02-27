import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from langchain_community.document_loaders import CSVLoader, TextLoader, DropboxLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).resolve().parent

def load_documents():
    print("Loading data...")
    loader = CSVLoader(str(ROOT / "articles/imdb_top_1000.csv"), encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    for i, doc in enumerate(documents[:3]):
        print(f"Document {i}: {doc.metadata['source']} - {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

if __name__ == "__main__":
    load_documents()


