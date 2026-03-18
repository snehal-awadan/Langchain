"""
It lets you load multiple documents from a directory (folder) of files.

    1) **/*.txt → All .txt files
    2) *.pdf → all .pdf
    3) data/*.csv → all csv files in data/ folder
    4) **/* → all files (any type, all folder)

"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    # path of the folder:
    path = '/home/snehal/Documents/NWDAF_DOCS',

    # which type of file wants to load from folder
    glob = '*.pdf',

    # which type of loader (based on type of files)
    loader_cls = PyPDFLoader 
)

docs = loader.load()

print(len(docs)) # will return the total number of pages of the load pdf

# to see first page of the first pdf:
print(docs[0].page_content)


