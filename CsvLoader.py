'''
It is used to load  CSV files into langchain document objects - one per row, by default.
'''

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='/home/snehal/AI_ML/NWDAF/Congestion_prediction/Dataset.csv')

data = loader.load()
print(data[0])
print(len(data))