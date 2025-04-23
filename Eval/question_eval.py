from openai import OpenAI
import pandas as pd
import re
import json
from tqdm import tqdm
from prompts import question_eval_prompt

server_url = 'http://127.0.0.1:1234/v1'
model_id = 'gemma-3-4b-it'
client = OpenAI(base_url=server_url, api_key="lm-studio")

df = pd.read_csv('questions.csv')

question_evals = []
for idx, row in tqdm(df.iterrows()):
    eval = {}
    for col in df.columns:
        eval[col] = row[col]
    question = row['question']
    chunk = row['chunk']
    response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": question_eval_prompt.format(
                        context=chunk,
                        question=question
                    )
                }
            ],
            temperature=0.6,
        )
    response = response.choices[0].message.content
    pattern = r'(relevance_rating|groundness_rating|standalone_rating):\s*(\d+)'
    ratings = {key: int(value) for key, value in re.findall(pattern, response)}
    for key in ratings.keys():
        eval[key] = ratings[key]
    question_evals.append(eval)
df = pd.DataFrame(question_evals)
df.to_csv('question-evals.csv')