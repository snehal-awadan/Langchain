'''

StructuredOutputParser helps you tell the LLM exactly what fields you want in the output and then extracts those fields from the response. You can specify the fields you want in the output and the parser will extract those fields from the response. This is useful when you want to extract specific information from the response and you don't want to deal with the entire response.

'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import ResponseSchema

model = ChatOllama(model='llama3')


# create shema:
schema = [
    ResponseSchema(name = 'fact_1', description = 'about the topic'),
    ResponseSchema(name = 'fact_2', description = 'about the topic'),
    ResponseSchema(name = 'fact_3', description = 'about the topic'),
]

parser = StructuredOutputParser.from_response_schema(schema)

# create template:

template = PromptTemplate(
    template = 'Give 3 fact about the {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instruction()})

chain = template | model | parser

result = chain.invoke({'topic': 'Moon'})

