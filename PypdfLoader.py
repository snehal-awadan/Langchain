"""
It is used to load the content from PDF files and convert each page into document object.
Limitation: It uses the pypdf library from the hood. - Not great with scanned PDF's/complex layouts.
"""

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/home/snehal/Downloads/Learning_to_Optimize__Edge_Based_Graph_Neural_Networks_Trained_on_MILP_Optimized_Routing_Paths-1-2.pdf')

docs  = loader.load()
# print(docs)
print(len(docs))

# extract 1st page:
print(docs[0].page_content)