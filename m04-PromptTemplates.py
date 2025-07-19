from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = PromptTemplate(template="Translate the following from English into {language}. {text}", input_variables=['language', 'text'])
prompt = prompt_template.invoke({"language": "French", "text": "Hi! My name is Mohammad Yehya Hayati. I am an employee at the prestigious Systems Limited."})
print(model.invoke(prompt).content)

# Prompt templates are majorly used for dynamic prompts. However, in this simple example, the model doesnt remember the old messages.