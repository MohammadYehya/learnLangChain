from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="The name of the person.")
    age: int = Field(description="The age of the person.")
    interests: list[str] = Field(description="The list of interests of the person")
    company_name: Optional[str] = Field(description="The company that the person works in.", default=None)

model = ChatOpenAI(model = "gpt-4o-mini")
structured_model = model.with_structured_output(Person)

result = structured_model.invoke("Hi! I am Mohammad Yehya Hayati, 22 years old, and I work in Systems Limited! I love programming and learning new stuff!")
print(result)

# Chat Models output textual data, meaning unstructured data. Some models have the option to return structured data, which is done by the above example.