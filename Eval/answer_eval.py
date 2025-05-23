from openai import OpenAI
import pandas as pd
import os
import csv
from tqdm import tqdm
from prompts import answer_eval

def write_dict_to_csv(file_path, dict_data, fieldnames):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerow(dict_data)

server_url = 'http://127.0.0.1:1234/v1'
model_id = 'gemma-3-4b-it'
client = OpenAI(base_url=server_url, api_key="lm-studio")

df = pd.read_csv('../generations.csv')
fieldnames = list(df.columns) + ['scores']

eval_df = pd.read_csv('evals.csv')

for idx, row in tqdm(df.iterrows()):
    row_dict = row.to_dict()
    question = row['question']
    collection = row['collection']
    limit = row['limit']
    model = row['model']
    answer = row['answer']
    subset = eval_df[(eval_df['question'] == question) & (eval_df['collection'] == collection) & (eval_df['limit'] == limit) & (eval_df['model'] == eval_df['model'])]
    if subset.shape[0] > 0:
        continue
    chunks = eval(row['context'])
    context = ""
    for i in range(len(chunks)):
        context += f"Chunk : {i+1}\n"
        context += chunks[i]
        context += "\n"
    response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": answer_eval.format(
                        question=question,
                        context=context,
                        answer=answer
                    )
                }
            ],
            temperature=0.5,
        )
    response = response.choices[0].message.content
    row_dict['scores'] = response
    write_dict_to_csv('evals.csv', row_dict, fieldnames)