from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# initialize the model:
model = ChatOllama(model = 'llama3')

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

# create retrievalQAchain:
qa_chain = RetrievalQA.from_chain_type(llm = model,
                                       retriever = retriever)


# Manually pass retrived text to llm:
query = 'explain the uploaded text '
answer = model.invoke(query)

#print the answer:
print('Answer: ', answer.content)