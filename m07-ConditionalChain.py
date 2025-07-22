from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="Sentiment value of the feedback.")
parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(template="Classify the sentiment of the following feedback.\n {feedback}\n {format_instruction}", input_variables=['feedback'], partial_variables={'format_instruction':parser2.get_format_instructions()})
classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(template='Write an appropriate response to this positive feedback \n {feedback}', input_variables=['feedback'])
prompt3 = PromptTemplate(template='Write an email to customer support about this negative feedback from a customer.\n {feedback}', input_variables=['feedback'])
branch_chain=RunnableBranch((lambda x:x.sentiment == 'Positive', prompt2 | model | parser), (lambda x:x.sentiment == 'Negative', prompt3 | model | parser), RunnableLambda(lambda x: "Could not find Sentiment"))
chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': "This is one of the best phones if I was in Hell because this is FIRE!!!."})
print(result)