"""
Semantic splitter divides text based on meaning, not just size or structure.

Instead of splitting by:
    length
    sentences
    sections

It checks:
    “Does this part of text talk about the same idea?”

If yes → keep together
If meaning changes → split

For example,
Text:
    AI is transforming industries. It is used in healthcare and finance.
    Cricket is a popular sport in India.

After semantic splitting:
    Chunk 1 → AI-related content
    Chunk 2 → Cricket-related content

Use when:

    content has multiple topics mixed together
    high accuracy is required
    building advanced RAG systems

"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type = 'standard_deviation', # It tells SemanticChunker: Use statistical variation (standard deviation) to decide where to split the text.
    breakpoint_threshold_amount = 0.006   # How sensitive the splitter is to topic change
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricke league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happens, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.

"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
