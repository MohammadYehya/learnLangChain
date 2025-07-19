from dotenv import load_dotenv
from langchain_openai import OpenAI
load_dotenv()

llm = OpenAI(model="gpt-4o-mini")
result = llm.invoke("Tell me the Capital of Pakistan")
print(result)

# LLMs are now outdated and LangChain suggests to use ChatModels