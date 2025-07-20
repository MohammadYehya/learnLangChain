from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Other models do no support structured outputs, so we use Output Parsers to get the output.

model = ChatOpenAI(model = 'gpt-4o-mini')

template1 = PromptTemplate(template="Write a detailed report on {topic}.", input_variables=['topic'])
template2 = PromptTemplate(template="Write a 5 line summary on the following text. \n {text}", input_variables=['text'])

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'Quantum Computing'})
print(result)
# StrOutputParser basically returns result.content