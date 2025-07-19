from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print('---\nNOT USING CHAT HISTORY!\n---')
user = "What is greater 10 or 5?"
print(user)

result = model.invoke(user).content
print(result)

user = "Now multiply the greater number with 10."
print(user)

result = model.invoke(user).content
print(result)
# ^ Doesnt store chat history


print('---\nUSING CHAT HISTORY!\n---')
chat_history = [SystemMessage(content = "You a a great mathematician.")]
user = "What is greater 10 or 5?"
print(user)
chat_history.append(HumanMessage(user))

result = model.invoke(chat_history).content
print(result)
chat_history.append(AIMessage(result))

user = "Now multiply the greater number with 10."
print(user)
chat_history.append(HumanMessage(user))

result = model.invoke(chat_history).content
print(result)
chat_history.append(AIMessage(result))

# Since models are stateless, they dont hold previous chat messages. Therefore, we use a list to store all messages and send that to the model
# AIMessage, HumanMessage, and SystemMessage are used to indicate the role of the person sending the message.