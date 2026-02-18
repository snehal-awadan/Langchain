'''
StrOutput Parser is the simplest output in langchain. It is used to parse the output of LLM and return it as a plain string.
'''

'''
Problem Statement: On basis of topic given the 1st LLM will give the detail explaination about it and then 2nd LLM will give the summary of that explaination.
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model='llama3')

template1 = PromptTemplate(
    input_variables = ['topic'],
    template = "Explain about the topic in detail : {topic}"
)

template2 = PromptTemplate(
    input_variables = ['text'],
    template = "Write 5 line summary of the following text : {text}"
)

# parser:
parser = StrOutputParser()

# Create chain:
chain = template1 | model | parser | template2 | model | parser

# run the chain:
result = chain.invoke( {'topic': 'Artificial Intelligence'} )


print(result[template2])