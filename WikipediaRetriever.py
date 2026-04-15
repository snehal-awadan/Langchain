'''
- A Wikipedia retriever is a retriever that queries the Wikipedia API to fetch relevant content for a given query.

- How it worked:
    - You give it a query.
    - It sends the query to Wikipedia's API
    - It retrieves the most relevant articles
    - It returns them as LangChain Document objects.
    
'''
from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(
    top_k_results=2,
    lang= 'en'
)

# define the query:
query = 'the geopolitical history of India and Pakistan from the perspective of a chinese'

docs = retriver.invoke(query)

# print the docs:
for i, doc in enumerate (docs):
    print(f'\n--- Result {i+1} ---')
    print(f'content:\n {doc.page_content}---')