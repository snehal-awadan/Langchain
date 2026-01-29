from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import streamlit as st

st.header("Research Assistant")

# initialize the ollama model:
model = ChatOllama(model = 'llama3',
                   temperature=0.3)

paper_input = st.selectbox('Select Research Paper',
                           options=['Attention is all you need', 
                                    'BERT-pre-training of Deep Bidirectional Transformers',
                                    'GPT-3: Language Models are Few-Shot Learners',
                                    'Diffusion models beat GANs on image synthesis'])


style_input = st.selectbox("Select explaination style", ['Beginner-friendly', 
                                                         'Technical', 
                                                         'Core-oriented', 
                                                         'Mathematical'])

length_input = st.selectbox('Select explaination length', ['Short (1-2 paragraphs)',
                                                           'Medium (3-5 paragraphs)',
                                                           'Long (detailed explanation)'])
# Create template:

template = PromptTemplate(
    template = """ 

Please summarize the research paper titled "{paper_input} with the following specifications:
Explaination Style: {style_input}
Explaination Length: {length_input}
1. Mathematical details:
    - Include relevant mathematical equations and derivations if present in the paper,
    - Explain the mathematical concept using simple, intuition code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
    - Provide examples that connect the paper's concepts to everyday experiences.
If certain information is not available in the paper, respond with :"Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables = ['paper_input', 'style_input', 'length_input']
)

# fill the placeholders:

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write("Research Summary:")
    st.write(result.content)
