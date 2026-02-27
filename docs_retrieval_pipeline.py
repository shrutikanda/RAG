import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

persist_directory = str(ROOT / "db/chroma_db")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(collection_name="documents",
             embedding_function=embedding_model, 
             persist_directory=persist_directory,
             collection_metadata={"hnsw:space": "cosine"})

query = "which island does SpaceX lease for its launches in the Pacific?"

#retriever = db.as_retriever(search_kwargs={"k": 3})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3})

retriever_results = retriever.invoke(query)

print("Retriever results:")
for i, doc in enumerate(retriever_results, 1):
    print(f"Result {i}: {doc.page_content[:500]}...")
    print(f"metadata: {doc.metadata}")
    print("---")


combine_input = f"""Based on the following documents, please answer this Question: {query}

Domuments:{chr(10).join([doc.page_content for doc in retriever_results])}

Please provide a concise answer based on the above documents. If you don't know the answer, say you don't know."""

model = ChatOpenAI(model="gpt-4")
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on retrieved documents."),
    HumanMessage(content=combine_input)  
]

model_response = model.invoke(messages)


print("Model response:")
print(model_response.content)
