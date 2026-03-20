'''
It used to load and extract text content from webpage (URL)

It uses BeautifulSoup under the hood to parse HTML and extract visible text.
(BeautifulSoup is a Python library used to parse HTML and extract data from web pages.)

When to use:
    For blogs, news articles, or public websites where the content is primarily text-based and static.

Limitation:
    - Doesn't handle JavaScript-heavy pages well (use SeleniumURLLoader for that)
    - Loads only static content (What's in the HTML, not what loads after the page renders)
'''

from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

'''
We can also add multiple url's with the help of list.
'''

url = 'https://www.amazon.in/One94Store-Creative-Engraved-Decoration-Birthday/dp/B0D9XP8FDD/?_encoding=UTF8&pd_rd_w=gnbZv&content-id=amzn1.sym.5c71aec9-f305-470c-a5ee-954c638d1aa3&pf_rd_p=5c71aec9-f305-470c-a5ee-954c638d1aa3&pf_rd_r=7P4SFBPNJ06FP2PG1XM7&pd_rd_wg=nfwVs&pd_rd_r=16a27913-b7ac-4fc4-9e32-95cba7d86b61&ref_=pd_hp_d_btf_PB&th=1'

loader = WebBaseLoader(url)
docs = loader.load()

# print(len(docs))
# print(docs[0].page_content)

#################################################################################################33

# we can also ask question based on the url define:
model = ChatOllama(model =  'llama3')
parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables = ['question','text'],
    template = "Answer the following question \n {question} from the following text  - \n {text}"
)

chain = prompt | model | parser

result = chain.invoke({'question' : 'About this product?', 'text': 'docs[0]'})
print(result)
