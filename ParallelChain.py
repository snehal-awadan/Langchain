# On created report we will generate Notes & Quix simultanously.

'''
            +---------------------------+            
            | Parallel<notes,quiz>Input |            
            +---------------------------+            
                 **               **                 
              ***                   ***                 
+----------------+                +----------------+ 
|PromptTemplate_1|               |PromptTemplate_2 | 
+----------------+                +----------------+ 
          *                               *          
          *                               *          
  +------------+                    +------------+   
  |  MODEl _1  |                    |  MODEl _2  |   
  +------------+                    +------------+   
          *                               *          
          *                               *          
+-----------------+              +-----------------+ 
|     Parser      |              |      Parser     | 
+-----------------+              +-----------------+ 
                 **               **                 
                   ***         ***                   
           +----------------------------+            
           | Parallel<notes,quiz>Output |            
           +----------------------------+            
                          *                          
                          *                          
                 +----------------+                  
                 | PromptTemplate_3 |                  
                 +----------------+                  
                          *                          
                          *                          
                   +------------+                    
                   |   LLM_1/2    |                    
                   +------------+                    
                          *                          
                          *                          
                +-----------------+                  
                |      Parser     |                  
                +-----------------+                  
                          *                          
                          *                          
              +-----------------------+              
              |        Output         |              
              +-----------------------+              

'''

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


model_1 = ChatOllama(model='llama3')
model_2 = ChatOllama(model='llama3')

prompt_1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variable = ['text'] 
)

prompt_2 = PromptTemplate(
    template = 'Generate 2 short question answers from the following text \n {text}',
    input_variable = ['text'] 
)

# merge
prompt_3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -->  {notes} and quiz --> {quiz}',
    input_variable = ['notes', 'quiz'] 
)

# parser:
parser = StrOutputParser()

# parallel chains:
parallel_chains = RunnableParallel({
    'notes': prompt_1 | model_1 | parser,
    'quiz': prompt_2 | model_2 | parser
})

# merge chain:
merge_chain = prompt_3 | model_1 | parser

chain = parallel_chains | merge_chain 


# input text:
text = """

Support Vector Machines (SVM) are powerful, supervised machine learning models used for classification, regression, and outlier detection by finding the optimal hyperplane that maximizes the margin between data classes. They are effective in high-dimensional spaces, using a subset of training points called support vectors.

Key Concepts and Features
    Optimal Hyperplane: The algorithm finds the best line or boundary (hyperplane) that separates classes with the maximum margin (distance between the closest data points).
    Kernel Trick: SVMs can handle non-linear data by mapping inputs into higher-dimensional feature spaces, allowing for linear separation in that space.
    Support Vectors: These are the data points nearest to the hyperplane, which define the margin and, consequently, the model's structure.
    Applications: Commonly used for image recognition, text classification, and bioinformatics.

    Advantages and Limitations

    Pros: Highly accurate, memory-efficient (only uses support vectors), and versatile due to various kernel functions.
    Cons: Not suitable for large datasets due to high training time, sensitive to noise, and requires careful feature scaling. 

Common SVM Implementations

    SVC (Support Vector Classification): Used for binary and multi-class classification.
    SVR (Support Vector Regression): Used for regression tasks.
    LinearSVC: Optimized for linear classification problems. 
    
"""
result = chain.invoke({'text': text})

print(result)

# to visualize the chain:
chain.get_graph().print_ascii()