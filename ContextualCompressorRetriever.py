'''
- It improves retrieval quality by compressing documents after retrieval by keeping only the relevant content based on the user's query.

- How it works:
    - Base retriever retrieves N documents.
    - A compressor (usually an LLM) is applied to each document.
    - The compressor keeps only the parts relevant to the query
    - Irrelevant content is discarded.

- When to use:
    - Your docs are long and contain mixed information.
    - You want to reduce context length for LLM's
    - You need to to improve answer accuracy in RAG pipelines. 
'''

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor 
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings


docs = [
    Document(page_content=(
        """The grand canyon is one the most visited natural wonders in the world,
         Photosynthesis is the process by which green plants convert sunlight into energy.
          Millions of tourists travel to see it every year. The rocks data back millions of years. """
    ), metadata = {'source': 'Doc1'}),

        Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
         The chlorophyll in plant cell captures sunlight during photosynthesis.
          Knights wore armor made of metal. Siege weapons were often used to breach castle walls. """
    ), metadata = {'source': 'Doc2'})
]

# create vectore store:
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, 
                                   embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs = {'k' : 5})

# set up the compression using LLM:
llm = ChatOllama(model = 'llama3')
compressor = LLMChainExtractor.from_llm(llm)

# create the contextual compression retriever:
compression_retriever = ContextualCompressionRetriever(
    base_retriever = base_retriever,
    base_compressor = compressor
)

query = 'what is photosynthesis?'
compress_result = compression_retriever.invoke(query)

for i, doc in enumerate(compress_result):
    print(f'\n --- Result {i+1} ---')
    print(doc.page_content)