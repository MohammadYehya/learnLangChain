from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions = 32)
result = embedding.embed_query("Islamabad is the capital of Pakistan.")
print(result)


documents = [
    "Islamabad is the capital of Pakistan.",
    "Delhi is the capital of India."
]
result = embedding.embed_documents(documents)
print(result)

# Embedding models are used to calculate Vector Embeddings according to a specific model, which can be used for semantic searches and RAG based solutions.