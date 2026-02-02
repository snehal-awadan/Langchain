# It is a way to define a dictionary with specific keys and value types.

from langchain_ollama import ChatOllama
from typing import TypedDict

# load model:
model = ChatOllama(model='llama3')

# define the schema of the structured output:
class Review(TypedDict):
    summary: str
    sentiment: str

model_review = model.with_structured_output(Review)

result = model_review.invoke("The movie was fantastic! I really enjoyed the plot and the characters were well developed.")
print(result)


