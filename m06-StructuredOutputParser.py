from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')

schema = [
    ResponseSchema(name='fact1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template1 = PromptTemplate(template="Give 3 facts about {topic}.\n {format_instruction}", input_variables=['topic'], partial_variables={'format_instruction':parser.get_format_instructions()})
chain = template1 | model | parser
result = chain.invoke({'topic':'Osmium'})
print(result)
# StructuredOutputParser allows you to specify the Json schema but not validate it