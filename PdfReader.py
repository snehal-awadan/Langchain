from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#load the document:
loader = TextLoader('/home/snehal/Documents/chatbot_speech.txt')
document = loader.load()

# split the text into smaller chunks:
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)

docs = text_splitter.split_documents(document)

# convert text into embeddings:
vectorestore = FAISS.from_documents(docs, embeddings)

# create a retriever:
retriever = vectorestore.as_retriever()

# Manually retrive relevant docs:
query = 'what is the model names used?'
retrieved_docs = retriever.get_relevant_documents(query)

# Combine retrieved text into a single prompt:
retrieved_text = '/n'.join(['doc.page_content for doc in retrived_docs'])

# initialize the model:
model = ChatOllama(model = 'llama3')
                        
# Manually pass retrived text to llm:
prompt = f'based on the following text, answer the question: {query} \n\n {retrieved_text}'
answer = model.invoke(prompt)

#print the answer:
print('Answer: ', answer.content)