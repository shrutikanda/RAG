
from anyio import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

def main():
    print("Hello from rag!")
    
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25


    # articles available here: {add GitHub repo}
    documents = SimpleDirectoryReader("articles").load_data()
    print(len(documents))
    # store docs into vector DB
    index = VectorStoreIndex.from_documents(documents)

    
    # set number of docs to retreive
    top_k = 3

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )    
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    # query documents
    query = "What is fat-tailedness?"
    response = query_engine.query(query)
    print(response) 
    # reformat response
    context = "Context:\n"
    for i in range(top_k):
        context = context + response.source_nodes[i].text + "\n\n"

    print(context)    
    
 

  
     

if __name__ == "__main__":
    main()
