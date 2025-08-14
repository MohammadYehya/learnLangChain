from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# A retriever is determined by which data source it works with and what search strategy they use (MMR, Multi-Query, etc)

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={'k':2})
query = "What are embeddings used for?"
results = retriever.invoke(query)
print([result.page_content for result in results])

# This regular retriever works the same way as vectorstore.similarity_search, but the advantage of a retriever is that it can incorporate different search strategies