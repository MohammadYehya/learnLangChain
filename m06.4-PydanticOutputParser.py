from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')

class Person(BaseModel):
    name: str = Field(description="Name of the person.")
    age: int = Field(gt=18, description="Age of the person.")
    city: str = Field(description="Name of the city the person is living in.")
parser = PydanticOutputParser(pydantic_object=Person)
template1 = PromptTemplate(template="Generate the name, age, and city of a fictional {nationality} person.\n {format_instruction}", input_variables=['nationality'], partial_variables={'format_instruction': parser.get_format_instructions()})
chain = template1 | model | parser
result = chain.invoke({'nationality':'Pakistani'})
print(result)
# PydanticOutputParser allows you to perform data validation using pydantic classes