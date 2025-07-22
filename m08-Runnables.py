from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate(template="Write a joke about {topic}", input_variables=['topic'])
model = ChatOpenAI(model = "gpt-4o-mini")
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser) # Same as chain = prompt | model | parser
# print(chain.invoke({'topic':'Black Hole'}))
# RunnableSequence is the standard runnable (| operator)

prompt2 = PromptTemplate(template="Write an interesting fact about {topic}", input_variables=['topic'])
chain = RunnableParallel({'Joke': prompt | model | parser, 'Fact': prompt2 | model | parser})
# print(chain.invoke({'topic':'Platinum'}))
# RunnableParallel allows us to run Runnables in parallel

chain = RunnablePassthrough()
# print(chain.invoke(2))
# RunnablePassthrough justs outputs its input. It is useful when paired with other Runnables like Parallel or Branch

chain = RunnableBranch((lambda x:x['topic'] == "Chemistry", prompt2 | model | parser), (lambda x:x['topic'] == "Computers", prompt | model | parser), RunnablePassthrough())
# print(chain.invoke({'topic':'Chemistry'}))
# print(chain.invoke({'topic':'Computers'}))
# print(chain.invoke({'topic':'Random'}))
# RunnableBranch allows us to conditionally run Runnables

chain = prompt | model | parser | RunnableParallel({'Joke': RunnablePassthrough(), 'Count': RunnableLambda(lambda x: len(x.split()))})
# print(chain.invoke({'topic': 'AI Models'}))
# RunnableLambdas allow us to use python functions as a Runnable