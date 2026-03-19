"""
1) load() : loads all documents from the source at once and returns them as a complete list.
All documents → loaded immediately → returned together
    Best when,
        - The number of files are small
        - you want everything loaded upfront.

2) lazy_load() : loads documents one by one only when needed instead of loading everything at once.
Documents → loaded gradually → saves memory and time
    Best when,
        - Dealing with large document/lots of files.
        - Want to stream processing (e.g. Chunking, embedding) without using lots of memory 
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

# using load()
'''
Will display all the document at a same time while taking time to load the document in the start.
'''
docs = loader.load()

for document in docs:
    print(document.metadata)


# using lazyload():
'''
Will stream all the document present in the folder one by one in the terminal.
'''
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)