from openai import OpenAI
from llama_cpp import Llama
import pandas as pd
from qdrant_client import QdrantClient
import csv
import os
from tqdm import tqdm
import re
from Eval.prompts import qa_prompt

def write_dict_to_csv(file_path, dict_data, fieldnames):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerow(dict_data)

server_url = 'http://127.0.0.1:1234/v1'
client = OpenAI(base_url=server_url, api_key="lm-studio")
models = ['gemma-3-1b-it']

df = pd.read_csv('Eval/question-evals.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

df = df[
    (df['relevance_rating'] == 5) &
    (df['groundness_rating'] == 5) &
    (df['standalone_rating'] == 5)
]

fieldnames = list(df.columns) + ['limit', 'collection', 'context', 'ctx_length', 'answer', 'model']

minilm_model_path = 'C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf'
embed_model = Llama(minilm_model_path, embedding=True, verbose=False)

collections = ['semantic', 'char', 'token', 'recursive', 'optimized-semantic', 'optimized-char', 'optimized-token', 'optimized-recursive']
limits = [3, 5]

generations_df = pd.read_csv('generations.csv')
generations_df.drop(columns=['Unnamed: 0'], inplace=True)

qdrant_client = QdrantClient(path="rag_eval")

for idx, row in df.iterrows():
    question = row['question']
    query_vector = embed_model.create_embedding(question)['data'][0]['embedding']
    for limit in limits:
        for collection in collections:
            generated = {}
            for col in df.columns:
                generated[col] = row[col]
            generated['limit'] = limit
            generated['collection'] = collection
            search_result = qdrant_client.search(collection_name=collection, query_vector=query_vector, limit=limit)
            context = []
            for result in search_result:
                context.append(result.payload['text'])
            context = [str(chunk) for chunk in context]
            generated['context'] = context
            context = '\n'.join(context)
            generated['ctx_length'] = len(context)
            for model in models:
                subset = generations_df[(generations_df['question'] == question) & (generations_df['limit'] == limit) & (generations_df['collection'] == collection) & (generations_df['model'] == model)]
                if subset.shape[0] > 0:
                    continue
                response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": """You are an helpful AI assistant. Given a question and related information as context,
                                you have to help the user by answering the question based on the context"""
                            },
                            {
                                "role": "user",
                                "content": qa_prompt.format(
                                    question = question,
                                    context = context
                                )
                            }
                        ],
                        tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "QuestionAnswering",
                                    "parameters": {
                                        "type": "object",
                                        "title": "QuestionAnswering",
                                        "properties": {
                                            "answer": {
                                                "title": "Answer",
                                                "type": "string",
                                                "description": "A detailed and comprehensive answer based solely on the provided context that fully addresses the question."
                                            }
                                        },
                                        "required": [
                                            "answer"
                                        ]
                                    }
                                }
                            }
                        ],
                        tool_choice={"type": "function", "function": {"name": "QuestionAnswering"}},
                        temperature=0.6,
                    )
                response = response.choices[0].message
                content = response.content
                pattern = r'(?:"answer"\s*:\s*"([^"]*)"|“answer”\s*:\s*“([^”]*)”|answer\s*:\s*([^"\n\r]*))'
                try:
                    if content:
                        match = re.findall(pattern, content)[0]
                        generated['answer'] = match
                        generated['model'] = model
                    else:
                        response = response.tool_calls[0].function.arguments
                        match = re.findall(pattern, response)[0]
                        generated['answer'] = match
                        generated['model'] = model
                    write_dict_to_csv('generations.csv', generated, fieldnames)
                except:
                    print(response)
                    print("content is : ")
                    print(content)