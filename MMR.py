'''

Core idea: 
"How can we pick results that are not only relevant to the query but also difficult from each other."

MMR is search strategy used in retriever designed to reduce redundancy in the retrieved results while maintaining high relevance to the query.

Problem without MMR:

When you search in vector DB:
    You may get results like:
    - Chunk 1 → IPL is a cricket league
    - Chunk 2 → IPL is popular in India
    - Chunk 3 → IPL teams and players

Problem:
    - All chunks are too similar
    - You miss other useful information

What MMR does, MMR says:

“Give me results that are relevant but not repetitive”
So output becomes:
    - Chunk 1 → IPL is a cricket league
    - Chunk 2 → History of IPL
    - Chunk 3 → IPL teams and players

Now results are:
    - Relevant 
    - Diverse 
'''

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Langchain make it easy to work with LLM's."),
    Document(page_content="Langchain is used to build LLM based application."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR help you get diverse results when doing similarity search."),
    Document(page_content="Langchain supports chroma, FAISS, Pinecone, and more.")
]

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

# create vectore store from docs:
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# enable MMR in the retriever:
retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        'k' : 3,         # top results
        'lambda_mult':1  # relevance diversity balance
    }
)

query = 'what is langchian?'
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f'\n ---- Result {i+1} ----')
    print(doc.page_content)