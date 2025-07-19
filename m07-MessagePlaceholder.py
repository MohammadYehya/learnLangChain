from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []
with open('m07-MessagePlaceholder.txt') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund?'})

print(prompt)

# Usually when a user opens an old chat, we need to load the old messages and add them to the chat_history. For this a MessagePlaceholder is used