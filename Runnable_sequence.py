'''
It is a sequencial chain of runnables in langchain that executes each step one after another,passing the output of one step as the input to the next.
Useful when you need to compose multiple sunnabels together in a structured workflow.
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

model = ChatOllama(model='llama3')

prompt1 = PromptTemplate(
    input_variables = ['topic'],
    template = "Generate joke for the given topic : {topic}"
)

prompt2 = PromptTemplate(
    input_variables = ['text'],
    template = "Explain the following joke : {text}"
)
parser = StrOutputParser()
chain = RunnableSequence(prompt1, model, parser, prompt2, model,parser)

print(chain.invoke({'topic' : 'AI'}))