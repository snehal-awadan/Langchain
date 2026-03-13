"""
RunnableLambda is used when you want to turn a normal Python function into a Runnable so it can be used inside a LangChain pipeline.

In simple words:

RunnableLambda lets you add your own Python logic inside a LangChain chain.
"""


'''
Q. Generate a detailed report on give topic.
if generate report words > 20 then summarize it. 
otherwise print it as it is (passthorugh)
'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch

model = ChatOllama(model =  'llama3')
parser = StrOutputParser()

prompt1 = PromptTemplate(
    input_variables = ['topic'],
    template = "Write a detailed report on : {topic}"
)

prompt2 = PromptTemplate(
    input_variables = ['text'],
    template = "Summarize the following text \n : {topic}"
)

# sequence chain:
report_gen_chain = RunnableSequence(prompt1, model, parser)

'''
Syntax for Branch:
        RunnableBranch(
            (if condition, Runnable),
            (else condition, Runnable),
            (default)
        )
'''

branch_chain = RunnableBranch(
    # if condition:
    (lambda x:len(x.split()) > 20, RunnableSequence(prompt2, model,parser)),
    # else or default condition:
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": 'Russia vs Ukraine'}))
