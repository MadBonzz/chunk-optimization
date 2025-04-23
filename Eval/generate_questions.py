from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import re
from prompts import question_prompt

server_url = 'http://127.0.0.1:1234/v1'
model_id = 'gemma-3-4b-it'
client = OpenAI(base_url=server_url, api_key="lm-studio")

df = pd.read_csv('context_rich_chunks.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

chunks = []
questions_df = None
if os.path.isfile('questions.csv'):
    questions_df = pd.read_csv('questions.csv')
    chunks = list(questions_df['chunk'].unique())

chunk_questions = []
try:
    for _, row in tqdm(df.iterrows()):
        chunk = row['chunk']
        if chunk in chunks:
            continue
        question_dict = {}
        for col in df.columns:
            question_dict[col] = row[col]
        response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": question_prompt.format(
                              context=chunk
                        )
                    }
                ],
                temperature=0.6,
            )
        response = response.choices[0].message.content
        questions = re.findall(r'"([^"]+)"', response)
        for question in questions:
            q_dict = question_dict.copy()
            q_dict['question'] = question
            chunk_questions.append(q_dict)
except Exception as e:
    print(f"Error: {e}")
    print(questions)
    if chunk_questions:
        chunk_df = pd.DataFrame(chunk_questions)
        if questions_df is not None:
            questions_df = pd.concat([questions_df, chunk_df], ignore_index=True)
        else:
            questions_df = chunk_df
        questions_df.to_csv('questions.csv', index=False)
    else:
        print("No questions generated to save.")
    raise  
if chunk_questions:
    chunk_df = pd.DataFrame(chunk_questions)
    if questions_df is not None:
        questions_df = pd.concat([questions_df, chunk_df], ignore_index=True)
    else:
        questions_df = chunk_df
    questions_df.to_csv('questions.csv', index=False)
