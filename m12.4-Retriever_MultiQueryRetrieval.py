from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

# mqr (Multi-Query Retrieval) -> Used to remove ambiguity in queries
# Takes the query and generates multiple semantically different version of that query using an LLm. Then performs retrieval using each sub-query, then combines and deduplicates the results.

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
vectorstore = Chroma.from_documents(documents=all_docs, embedding=OpenAIEmbeddings())
retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=ChatOpenAI())
results = retriever.invoke("How to improve energy levels and maintain balance?")
# This is an abigious query pertaining our docs. Energy is in many docs (H5, I1, I3, I4, I5) and balance exists in I1
print([result.page_content for result in results])