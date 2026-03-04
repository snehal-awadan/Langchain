import random
from abc import ABC, abstractmethod

class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass

class NakliLLM(Runnable):
    def __init__(self):
        #print('LLM created!')
        pass
    
        # as this class is inherited by runable so we have to add parent method (invoke) in it 
        # which will act as same as predict()
    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}

        
    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}

llm = NakliLLM()

print(llm.predict('what is the capital of India?'))  # any response from response_list



# Class which will act as a PromptTemplate:
class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

# predict method --> format 
    def Format(self, input_dict):
        return self.template.format(**input_dict)

# predict --> invoke
    def invoke(self, input_dict):
        return self.template.format(**input_dict)


class NaklistrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']
    

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    
    def invoke(self, input_data):
        '''
        From chain, it will first create prompt then this prompt give it to the llm to predict, then this predict is go to the parser to get only str as output. 
        '''
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data



# create template:
template = NakliPromptTemplate(
    template = 'Write a poem about {topic}',
    input_variables = ['topic']
)

prompt = template.Format({'topic' : 'India'})


template1 = NakliPromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables= ['topic']
)

template2 = NakliPromptTemplate(
    template = 'Explain the following joke {response}',
    input_variables= ['response']
)
llm = NakliLLM()

parser = NaklistrOutputParser()

# will give template , llm  and then parser as list 
chain = RunnableConnector([template, llm, parser])


chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])

print(chain1.invoke({'topic' : 'AI'}))

final_chain = RunnableConnector([chain1, chain2])

print(final_chain.invoke({'topic' : 'cricket'}))


# class which will act as a chain:
class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.Format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']
    
chain = NakliLLMChain(llm, template)
print(chain.run({'topic' : 'India'}))