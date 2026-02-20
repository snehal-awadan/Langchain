'''
It uses pydantic models to enforce schema validation when processing LLM responses.

It allow:
    - Strict schema enforcement (Ensures LLM responses follows a well defined structured)
    - Type safety (automaticaly converts LLM output into python objects)
    - Easy validation (Uses pydantic build-in validation to catch incorrect /missing data)
    - Seamless integration (Works well with other langchain components)
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Step 1: Define Pydantic model
class Person(BaseModel):
    name: str
    age: int = Field(gt = 18)  # greater than 18 -> constraint
    city: str

# Step 2: Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Step 3: Create prompt WITH format instructions
prompt = PromptTemplate(
    template="""
    Generate a random name, age and city of the {place} person.

    {format_instructions}
    """,
    input_variables=["place"],

    # It only instructs the LLM to follow the define format.
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# Step 4: Create model
model = ChatOllama(model="llama3")

# Step 5: Create chain
chain = prompt | model | parser

# Step 6: Invoke
result = chain.invoke({"place": "Indian"})

print(result)
