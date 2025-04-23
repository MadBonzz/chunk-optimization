import numpy as np
import pymupdf
from llama_cpp import  Llama
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import re
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from optimization.src.optimizer import ChunkOptimizer
from optimization.src.utils import find_scores

minilm_model_path = 'C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf'

embed_model = Llama(minilm_model_path, embedding=True, verbose=False)

documents = {}

# Document 1
insurance_act = 'insurance-information/Insurance Act,1938 - incorporating all amendments till 20212021-08-12.pdf'
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
policyholder_file = 'insurance-information/Draft IRDAI(Protection of Policyholders’ Interests and Allied Matters of Insurers) Regulations, 2024.pdf'
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
handbook_path = 'insurance-information/Life Insurance Handbook (English).pdf'
doc = pymupdf.open(handbook_path)
complete_text = ""
for page in doc.pages(2):
    text = page.get_text()
    complete_text += text
text = re.sub(r'\n\d+', '', complete_text)
text = text.replace('  ', '').replace('\n \n', '\n').replace('\n\n', '\n').replace('. \n', '').replace('•', '')
documents['handbook'] = text

class MyLocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.array(embed_model.create_embedding(text)['data'][0]['embedding']) for text in texts]

    def embed_query(self, text):
        return np.array(embed_model.create_embedding(text)['data'][0]['embedding'])
    
embeddings = MyLocalEmbeddings()
semantic_splitter = SemanticChunker(embeddings)

char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=2000,
    chunk_overlap=0,
    is_separator_regex=False
)

recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=2000,
    chunk_overlap=0
)

splitters = {'char' : char_splitter, 'token' : token_splitter, 'recursive' : recursive_splitter, 'semantic' : semantic_splitter}

chunk_embeddings = {}
for splitter in splitters.keys():
    split_embeddings = {}
    splits = []
    for doc in documents.keys():
        splits.extend(splitters[splitter].split_text(documents[doc]))
    split_embeddings['splits'] = splits
    split_embeddings['base_embeddings'] = {idx : {'Text' : x, 'Embedding' : np.array(embed_model.create_embedding(x)['data'][0]['embedding']).reshape(1, -1), 'can_join' : True} for idx, x in enumerate(splits)}
    split_embeddings['base_embeddings'] = find_scores(split_embeddings['base_embeddings'])
    chunk_embeddings[splitter] = split_embeddings

optimized_embeddings = {}
for chunk in chunk_embeddings.keys():
    optimizer = ChunkOptimizer(embed_model=embed_model)
    optimized_embeddings['optimized'+'-'+chunk] = optimizer.optimize_chunks(2000, 500, 0.7, base_embeddings=chunk_embeddings[chunk]['base_embeddings'])

split_dict = {}
for key in chunk_embeddings.keys():
    split_dict[key] = {}
    split_dict[key]['base_embeddings'] = chunk_embeddings[key]['base_embeddings']
for key in optimized_embeddings.keys():
    split_dict[key] = {}
    split_dict[key]['base_embeddings'] = optimized_embeddings[key]

print(split_dict.keys())

client = QdrantClient(path="rag_eval")

client.create_collection(collection_name='semantic', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='char', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='token', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='recursive', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='optimized-semantic', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='optimized-char', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='optimized-token', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.create_collection(collection_name='optimized-recursive', vectors_config=VectorParams(size=384, distance=Distance.COSINE))

# Create collection for semantic splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['semantic']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='semantic',
    wait=True,
    points=points
)

points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['char']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='char',
    wait=True,
    points=points
)

points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['token']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='token',
    wait=True,
    points=points
)

# Create collection for recursive splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['recursive']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='recursive',
    wait=True,
    points=points
)

#Create collection for optimized semantic splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['optimized-semantic']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='optimized-semantic',
    wait=True,
    points=points
)

#Create collection for optimized char splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['optimized-char']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='optimized-char',
    wait=True,
    points=points
)

#Create collection for optimized token splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['optimized-token']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='optimized-token',
    wait=True,
    points=points
)

#Create collection for optimized recursive splits
points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['Embedding'].flatten().tolist(),
    payload = {
      "text": chunk['Text']
    }
  )
  for chunk in split_dict['optimized-recursive']['base_embeddings'].values()]

operation_info = client.upsert(
    collection_name='optimized-recursive',
    wait=True,
    points=points
)