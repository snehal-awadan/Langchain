'''
- Sometimes a single query might not capture all the ways information is phrased in your document.

- for example,
Query : "How can I stay healthy?

could means,
    - What should I eat?
    - How often should I exercise?
    - How can I manage stress?


Solution,
    - Take your original query
    - Uses an LLM to generate multiple semantically different versions of that query.
    - Performs retrieval for each sub-query
    - Combines and duplicates the results.


- Core idea, It try to solve the ambiquiety of the query, and generate multiple related query out of it and generate the result.

'''

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

docs = [
    Document(page_content="Langchain make it easy to work with LLM's."),
    Document(page_content="Langchain is used to build LLM based application."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR help you get diverse results when doing similarity search."),
    Document(page_content="Langchain supports chroma, FAISS, Pinecone, and more.")
]

# create embedding:
vectorestore = FAISS.from_documents(documents = docs,
                                    embedding = embeddings)


# multi-query retriever:

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vectorestore.as_retriever(
        search_kwargs = {'k' : 5}),
        llm = ChatOllama(model = 'llama3')                                         
)

# Query:

query = 'what is langchain?'

similarity_result = multiquery_retriever.invoke(query)
multiquery_result = multiquery_retriever.invoke(query)

for i, doc in enumerate(multiquery_retriever):
    print(f'\n --- Result {i+1}--- ')
    print(doc.page_content)