"""
Text structure-based splitter divides text based on natural structure like paragraphs, sentences, or lines.

How it works:
Instead of blindly splitting by size, it looks at:
    - paragraphs (\n\n)
    - sentences (.)
    - lines (\n)
and splits accordingly.

Advantages:
    - Keeps sentences complete
    - Maintains basic meaning
    - More natural chunks

Dis-advantages:
    - Depends on proper formatting (if text has no clear paragraphs/sentences, it fails)
    - Chunk sizes can become uneven (some too large, some too small)
    - Still does not understand deep meaning (semantic context)
    - Important information may still get split across chunks
"""



from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.[1]

High-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., language models and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as such: "A lot of cutting-edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."[2][3]

Various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[4] Some companies, such as OpenAI, Google DeepMind and Meta,[5] aim to create artificial general intelligence (AGI) – AI that can complete virtually any cognitive task at least as well as a human. 
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 0
)

# perform text spliter:
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)