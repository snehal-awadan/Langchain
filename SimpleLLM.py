from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model = 'llama3')

# create template:

prompt = PromptTemplate(
    template = "Suggest a catchy blog title above {topic}",
    input_variables = ['topic'] 
)

# Define input:
topic = input('Enter a topic: ')

# format the prompt manually:
formatted_prompt = prompt.format(topic = topic)

# call the LLM:
blog_title = model.invoke(formatted_prompt)

# print the output:
print('Generated Blog Title: ', blog_title.content)