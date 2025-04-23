import pandas as pd
import pymupdf
from llama_cpp import  Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import re

embedding_model_path = 'C:/Users/shour/.cache/lm-studio/models/Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1/multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf'
embedding_model = Llama(model_path=embedding_model_path, embedding=True, verbose=False)

documents = {}

# Document 1
insurance_act = '../insurance-information/Insurance Act,1938 - incorporating all amendments till 20212021-08-12.pdf'
doc = pymupdf.open(insurance_act)
complete_text = ""
for page in doc.pages(12):
    text = page.get_text()
    text = text[2:]
    paras = text.split('\n \n \n1')[:-1]
    for para in paras:
        complete_text += para

text = re.sub(r'\d\*{1,3}', '', complete_text)
text = re.sub(r'\n\d+', '', text)
text = re.sub(r'\(\d+\)', '', text)
text = text.replace('*', '').replace('\n \n', '\n').replace('  ', ' ').replace('  ', ' ').replace('[', '').replace(']', '').replace(' \n', '\n').replace(' .', '.').replace('..', '.')
text = text.strip()
documents['insurance_act'] = text

# Document 2
policyholder_file = '../insurance-information/Draft IRDAI(Protection of Policyholders’ Interests and Allied Matters of Insurers) Regulations, 2024.pdf'
doc = pymupdf.open(policyholder_file)
complete_text = ""
for page in doc.pages(2):
    text = page.get_text()
    complete_text += text
complete_text = complete_text[235:]

text = re.sub(r'\d+\s*\|\s*P\s*a\s*g\s*e', '', complete_text)
text = re.sub(r'\(\d+\)', '', text)
text = re.sub(r'\n\d+', '', text)
text = text.replace('  ', '').replace('\n \n', '\n').replace('\n\n', '\n').replace('. \n', '').replace('*','').replace('__', '_')
documents['policyholder'] = text

# Document 3
handbook_path = '../insurance-information/Life Insurance Handbook (English).pdf'
doc = pymupdf.open(handbook_path)
complete_text = ""
for page in doc.pages(2):
    text = page.get_text()
    complete_text += text
text = re.sub(r'\n\d+', '', complete_text)
text = text.replace('  ', '').replace('\n \n', '\n').replace('\n\n', '\n').replace('. \n', '').replace('•', '')
documents['handbook'] = text

chunks = []
for size in [1000, 2000, 3000]:
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=size,
        chunk_overlap=0,
        is_separator_regex=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=size,
        chunk_overlap=0
    )
    splitters = {'char' : char_splitter, 'token' : token_splitter, 'recursive' : recursive_splitter}
    for doc in documents.keys():
        for splitter in splitters.keys():
            chunk_dict = {'size' : size, 'document' : doc, 'splitter' : splitter}
            splits = splitters[splitter].split_text(documents[doc])
            chunk_dict['splits'] = splits
            chunks.append(chunk_dict)

df = pd.DataFrame(chunks)
df.to_csv('chunks.csv')
