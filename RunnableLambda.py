'''
RunnableLambda is used when you want to turn a normal Python function into a Runnable so it can be used inside a LangChain pipeline.

In simple words:
RunnableLambda lets you add your own Python logic inside a LangChain chain.

                           / Passthrough --> Joke
                          /
                         /
Prompt --> LLM --> Parser 
                         \ 
                          \
                           \ Lambda --> No. of words in joke 

'''
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

model = ChatOllama(model =  'llama3')
parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables = ['topic'],
    template = "Generate joke for the given topic with funnay emoji's : {topic}"
)

# to count the words:
def word_count(text):
    return len(text.split())

# sequence part:
joke_gen_chain = RunnableSequence(prompt, model, parser)


# parallel part:
parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

final_result = """ {} \n  Word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
