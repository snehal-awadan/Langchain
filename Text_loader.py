'''
It reads plain text (.txt) files and convert them into langchain document object.

Ideal for loading chat logs, scraped text, transcript, code snippet, etc into the langchain pipeline.

But has limitation, it only works with .txt file.
'''

from langchain_community.document_loaders import TextLoader

loader = TextLoader("/home/snehal/Documents/DigitalTwin+genAi.txt",
                    encoding = 'utf-8')

docs = loader.load()
print(docs)
print(type(docs))   # list 
print(len(docs))    # 1

print(docs[0])