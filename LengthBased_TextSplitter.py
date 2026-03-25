'''

Length-based text splitter divides text based on size (number of characters or tokens).
It simply checks:
    how long the text is
    and splits it into fixed-size chunks

For example,
    Chunk size = 50 characters
    "Artificial Intelligence is transforming industries and changing the way we live and work."

After splitting:
    Chunk 1 → "Artificial Intelligence is transforming industr"
    Chunk 2 → "ies and changing the way we live and work."

Disadvantages:
    - It does not understand meaning (semantic context)
    - It can break sentences in the middle
    - It can split important information across chunks
    - Results may be less accurate in retrieval (RAG)
    - Context continuity may be lost

'''

from langchain_text_splitters import CharacterTextSplitter

text = """
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.[1]

High-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., language models and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as such: "A lot of cutting-edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."[2][3]

Various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[4] Some companies, such as OpenAI, Google DeepMind and Meta,[5] aim to create artificial general intelligence (AGI) – AI that can complete virtually any cognitive task at least as well as a human. 
"""


splitter = CharacterTextSplitter(
"""
Chunk overlap means repeating some part of the previous chunk in the next chunk.
When we split text, we might lose context between chunks. So overlap helps to:
    - maintain continuity
    - avoid loss of important information
"""
    chunk_size = 100,
    chunk_overlap = 0,

    separator=''
)

result = splitter.split_text(text)
print(result)

print()
######################################################################

# Work flow: Load pdf from source and use text splitter:

print('*'*97)
print(" WORK-FLOW-2 : Load pdf from source and use text splitter")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/home/snehal/Downloads/System Architecture for OBD.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 0,
    separator=''
)

result_1 = splitter.split_documents(docs)
print(result_1[0].page_content)