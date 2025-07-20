from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')

parser = JsonOutputParser()
template1 = PromptTemplate(template="Give me the name, age, and city of a fictional person.\n {format_instruction}", input_variables=[], partial_variables={'format_instruction':parser.get_format_instructions()})

chain = template1 | model | parser
result = chain.invoke({})
print(result)
# JsonOutputParser allows you to tell the ChatModel to output in Json format, but doesn't allow for a specified format.
