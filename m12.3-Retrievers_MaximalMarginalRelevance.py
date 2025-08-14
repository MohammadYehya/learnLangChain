from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# mmr (Maximal Marginal Relevance) -> Used for diversity in retrieval
# Tries to find all relevant docs but reducing relevancy between each relavant doc

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# Normal Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3, 
    }
)
print("Normal Retriever: ", [result.page_content for result in retriever.invoke("What is langchain?")])

# Retriever with MMR Strategy
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3, 
        "lambda_mult": 0.5 # lambda_mult is the relavance-diversity balance, and is required when using mmr, 1 -> Normal Similarity Search, 0 -> Very Diverse Results
    }
)
print("Retriever with MMR Strategy: ", [result.page_content for result in retriever.invoke("What is langchain?")])
