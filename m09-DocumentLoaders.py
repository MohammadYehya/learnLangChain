from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = ChatOpenAI(model = "gpt-4o-mini")
prompt = PromptTemplate(template="Give me the number in the following data. \n {data}", input_variables=['data'])
parser = StrOutputParser()
loader = TextLoader("DocumentLoaders.txt")
docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'data':docs[0].page_content}))
# TextLoader loads txt files
# There are many more Document loaders which can be found online on their docs
# DirectoryLoader is used to load a directory of files
# Also if the data is so big that it lags due to loading in memory, we can use lazy_load instead of load.