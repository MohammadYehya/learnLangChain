from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

print('[INFO]:\t\tRetrieving Video Transcript.')
transcript = " ".join(chunk.text for chunk in YouTubeTranscriptApi().fetch(video_id="KMZT1aRFZi4", languages=["en"]).snippets)
print('[INFO]:\t\tVideo Transcript Retreived.')

print('[INFO]:\t\tSplitting Transcript.')
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print('[INFO]:\t\tTranscript split into chunks.')

print('[INFO]:\t\tCreating ChromaDB Vector Store.')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(chunks, embeddings)
print('[INFO]:\t\tCreated ChromaDB Vector Store to hold Transcript Vector Embeddings.')

print('[INFO]:\t\tCreating Vector Store Retriever')
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
}) | prompt | llm | StrOutputParser()

print('[INFO]:\t\tExecuting LLM Chain.')
print(chain.invoke('Can you summarize the video'))