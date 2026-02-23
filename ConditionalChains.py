'''
On basis of feedback, analyze it as positive/negative/neutral and reply accordingly. 

                                Feedback
                                    |   
                                    |
                                 Analyze 
                                 /    \   
                                /      \
                               /        \
                        Positive       Negative

So based on nature of the feedback, generate EITHER thank you message OR Sorry message

'''
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel,RunnableBranch, RunnableLambda



model = ChatOllama(model='llama3')

# parser:
parser = StrOutputParser()

class  Feedback(BaseModel):
    sentiment : Literal ['Positive', 'Negative'] = Field(description='Give the sentiment of the feedabck ')
 
parser_2 = PydanticOutputParser(pydantic_object = Feedback)

# # Prompt : To check sentiment of the feedback
prompt_1 = PromptTemplate(
    template = 'Classify the sentiment of the following text into positive or negative \n {feedback} \n {format_instructions}',
    input_variable = ['feedback'],
    partial_variables = {'format_instructions':parser_2.get_format_instructions()} 
)

classifier_chain = prompt_1 | model | parser_2

# # only retrieve the sentiment with define format for condition reply
# result = classifier_chain.invoke({'feedback' : 'this is a terrible smartphone'}).sentiment
# print(result)


# Prompt : if feedback is positive
prompt_2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variable = ['feedback']
)

# Prompt : if feedback is NEgative
prompt_3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variable = ['feedback']
)


branch_chain = RunnableBranch(
    # if condition
    (lambda x:x['sentiment'] == 'positive', prompt_2 | model | parser),

    # else condition
    (lambda x:x['sentiment'] == 'negative', prompt_3 | model | parser),

    # otherwise/default
   # As this is not chain so we can't directly write lambda function,we have to run this function using RunnableLambda method 

    RunnableLambda(lambda x: 'could not find any sentiment may be its neutral')
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback" : 'this is a terrible smartphone'})