from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
from prompts import evaluation_prompt, rules, guidelines, examples

server_url = 'http://127.0.0.1:1234/v1'
model_id = 'gemma-3-4b-it'
client = OpenAI(base_url=server_url, api_key="lm-studio")

df = pd.read_csv('chunks.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

chunks = []
for idx, row in tqdm(df.iterrows()):
    splits = eval(row['splits'])
    lens = [len(chunk) for chunk in splits]
    splits = [chunk for chunk in splits if len(chunk) > 50]
    print(len(splits))
    for split in splits:
        chunk = {}
        for col in df.columns:
            if col == 'splits':
                continue
            else:
                chunk[col] = row[col]
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": evaluation_prompt.format(
                        rules=rules,
                        guidelines=guidelines,
                        examples=examples,
                        context=split
                    )
                }
            ],
            temperature=0.6,
        )
        response = response.choices[0].message.content
        match = re.search(r'final-Score:\s*([0-9]*\.?[0-9]+)', response)
        if match:
            final_score = float(match.group(1))
            chunk['score'] = final_score
            chunk['chunk'] = split
            if final_score >= 0.5:
                chunks.append(chunk)
print(len(chunks))
df = pd.DataFrame(chunks)
df.to_csv('context_rich_chunks.csv')