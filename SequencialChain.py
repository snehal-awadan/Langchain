# Application : Topic --> LLM_1 --> Report  --> LLM_2 --> Summary

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser


prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_varible = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a summary from the following text \n {text}',
    input_varible = ['text']
)

model = ChatOllama(model='llama3')

# parser:
parser = StrOutputParser()

# Chains:
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'moon'})

print(result)