'''
 IT is a special runnable primitive that simply returns the input as output without modifying it.
  
without RunnablePassthrough,
    i/p --> LLM --> o/p  (original input is gone)

with RunnablePassthrough,
            → LLM →
Input →                 → Final Output
            → Input →

'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough


model = ChatOllama(model =  'llama3')

prompt1 = PromptTemplate(
    input_variables = ['topic'],
    template = "Generate joke for the given topic : {topic}"
)

prompt2 = PromptTemplate(
    input_variables = ['text'],
    template = "Explain the following joke : {text}"
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2, model,parser)
    }
)

# connect both chains:
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic': 'cricket'}))