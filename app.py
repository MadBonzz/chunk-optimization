from llama_cpp import Llama
import pymupdf4llm
from langchain.embeddings.base import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from openai import OpenAI
import re

policy_path = 'Brochures\online-term-plan-plus-policy-contract.pdf'
embed_model_path = 'C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf'
llm_id = 'meta-llama-3.1-8b-instruct'
reranker_model_path = 'C:/Users/shour/.cache/lm-studio/models/gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q4_K_M.gguf'
policy_collection_name = 'policy'
reference_collection_name = 'references'
server_url = 'http://192.168.84.106:1234/v1'

def rerank_documents(query, retrieved_docs, model_path, top_k=5):
    model = Llama(
        model_path=model_path,
        embedding=True,
        verbose=False
    )
    
    scores = []
    
    for doc in retrieved_docs:
        formatted_input = f"[QRY] {query} [DOC] {doc}"
        try:
            result = model.eval(formatted_input)
            score = result[0] if isinstance(result, list) else result
        except:
            query_embedding = np.array(model.embed(f"[QRY] {query}"))
            doc_embedding = np.array(model.embed(f"[DOC] {doc}"))
            score = np.dot(query_embedding, doc_embedding.transpose()) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
        scores.append(score)
    
    doc_score_pairs = list(zip(retrieved_docs, scores))
    print(doc_score_pairs)
    
    ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in ranked_docs[:top_k]]

gen_client = OpenAI(base_url=server_url, api_key="lm-studio")

embed_model = Llama(model_path=embed_model_path, 
                    embedding=True,
                    verbose=False)

class MyLocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.array(embed_model.create_embedding(text)['data'][0]['embedding']) for text in texts]

    def embed_query(self, text):
        return np.array(embed_model.create_embedding(text)['data'][0]['embedding'])

embeddings = MyLocalEmbeddings()
splitter = SemanticChunker(embeddings)

md_text = pymupdf4llm.to_markdown(policy_path)

texts = splitter.split_text(md_text)

chunks = []
for i in range(len(texts)):
    chunk = dict()
    chunk['content']   = texts[i]
    chunk['length']    = len(texts[i])
    chunk['embedding'] = embed_model.create_embedding(texts[i])['data'][0]['embedding']
    chunks.append(chunk)

client = QdrantClient(path="insurance-reference")

try:
    if client.get_collection(policy_collection_name):
            client.delete_collection(collection_name=policy_collection_name)
except ValueError:
    print("Collection not found")

client.create_collection(
    collection_name=policy_collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['embedding'],
    payload = {
      "text": chunk['content']
    }
  )
  for chunk in chunks]

operation_info = client.upsert(
    collection_name=policy_collection_name,
    wait=True,
    points=points
)

question = input("Enter your question : ")

query_vector = embed_model.create_embedding(question)['data'][0]['embedding']
policy_result = client.query_points(
  collection_name=policy_collection_name,
  query=query_vector,
  limit=3
)


reference_result = client.query_points(
    collection_name=reference_collection_name,
    query=query_vector,
    limit=3
)

contexts = []

for context in policy_result.points:
    #print(context)
    contexts.append(context.payload['text'])

for context in reference_result.points:
    contexts.append(context.payload['text'])


query_maker = """
You would be provided a user question and relevant context from the user's policy document and reference documents about insurance.
Explain the answer to the user in a very simple and friendly language in detail. If the question is regarding a terminology or concept,
use examples to explain. Use very simple english, do not use complext language.
IF THE QUERY IS NOT RELATED TO INSURANCE, ANSWER WITH "NOT RELEVANT". DO NOT ANSWER ANY NON INSURANCE RELATED QUERIES.

The user query is : {query}

The relevant context from user's policy is : {context}

Provide the only the final answer starting as Answer : your answer

"""

reference_query = gen_client.chat.completions.create(
     model=llm_id,
     messages=[
        {
            "role": "system",
            "content": "You are an AI assistant that helps the user understand their insurance policy based on the provided context."
        },
        {
            "role": "user",
            "content": query_maker.format(
                query = question,
                context=contexts
            )
        }
    ],
    temperature=0.7,
)

#print(contexts)
print(reference_query.choices[0].message.content)