"""
Document structure-based splitter divides text based on sections like headings and titles to preserve context and improve retrieval in structured documents.

Instead of splitting by length and sentences 

It uses document elements, like:
    - headings (H1, H2, H3)
    - sections
    - chapters
    - titles

For example,
Suppose a document:
    # Introduction
    AI is transforming industries.

    # Applications
    AI is used in healthcare and finance.

After splitting:
    Chunk 1 → Introduction section
    Chunk 2 → Applications section 

When to use it?
    - documents have clear structure
    - PDFs
    - research papers
    - reports
    - documentation    
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# with Python class:

print('** with Python class:')

text = """

class Student:
    def __init__(self, name, age, grade):
        self.name =  name
        self.age =  age
        self.grade =  grade # grade is a float 

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >=6.0

Example usage:
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("the student is passing")
else:
    print("the student is not passing")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])  # everything related to class

print(chunks[1])  



print()

##############################################################################


print('** With Markdown:')


text_mark = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/xyz-abd/student-tracker.git

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=0,
)

# Perform the split
chunks_mark = splitter.split_text(text_mark)

print(len(chunks_mark))
print(chunks_mark[0])