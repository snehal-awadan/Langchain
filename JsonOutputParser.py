from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model='llama3')


parser = JsonOutputParser()

# create template:

template = PromptTemplate(
    template = "Give me the name, age, and city of the fictional person \n {format_instructions}",
    input_variables = [],
    partial_variables = {
        "format_instructions": parser.get_format_instructions()
    }
)

# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# instead of above 3 lines code, we can use chain:
chain = template | model | parser
final_result = chain.invoke({})

print(final_result)

print(type(final_result)) # must return <class 'dict'> ==> JSON

# as it is dict, we can access the specific values :
print(final_result["name"])
print(final_result["age"])
print(final_result["city"])