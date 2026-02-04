# Pydantic is a tool that helps you define and validate data in a clean, structured way.

# To install the pydantic lib: pip install pydantic

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'snehal'
    
    # Optional field:
    age : Optional[int] = None

    # email
    email : EmailStr

    # cgpa with range:
    cgpa : float = Field(gt = 0, lt = 11)

new_student = {'age': 25}
# new_student = {'name': 123}  # This will raise a validation error

# new_student =   {}  # this is also valid, name will take default value 'snehal' & age will be None

# new_student = {'age' : '25'}  # this is also valid, pydantic will convert string to int

new_student = {'name': 'John Doe', 'age': 22, 'email': 'abc@gmail.com', 'cgpa': 9}  # valid data

# create object:
student = Student(**new_student)

print(student)