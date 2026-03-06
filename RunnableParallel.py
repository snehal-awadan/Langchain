
'''
It allow multiple runnables to execute in parallel.
Each runnable receives the SAME input and process it independently, producing a dictionary of output.
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence


model = ChatOllama(model =  'llama3')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a tweet about a topic",
    input_variable = ['topic']
)

prompt2 = PromptTemplate(
    template = "Generate a linkdin post about {topic}",
    input_variable = ['topic']
)

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser),
        'linkdin':RunnableSequence(prompt2, model, parser)
    }
)


print(parallel_chain.invoke({'topic': 'AI'}))

# to visualize the chain:
parallel_chain.get_graph().print_ascii()