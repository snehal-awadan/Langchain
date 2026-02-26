from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate(
    template = 'Generate 2 interesting facts about {topic}',
    input_variable = ['topic']
)

model = ChatOllama(model='llama3')

# parser:
parser = StrOutputParser()

# chain:
chain = prompt | model | parser

final_result = chain.invoke({'topic': 'cricket'})
print(final_result)

# to visualize the chain:
chain.get_graph().print_ascii()