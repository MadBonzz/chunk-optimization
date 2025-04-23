import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def find_scores(split_embeddings):
    for i in range(len(split_embeddings.keys())):
        if i == len(split_embeddings.keys()) - 1:
            score = 0
        else:
            score = cosine_similarity(split_embeddings[i]['Embedding'], split_embeddings[i+1]['Embedding'])[0][0]        
        split_embeddings[i]['Score'] = score
    return split_embeddings

def combine_smaller(split_embeddings, min_chunk_size):
    to_remove = []
    for i in split_embeddings.keys():
        if len(split_embeddings[i]['Text']) < min_chunk_size:
            if((split_embeddings[i]['Score'] >= split_embeddings[max(i-1,0)]['Score']) or ((i-1) in to_remove)):
                split_embeddings[i+1]['Text'] += split_embeddings[i]['Text']
                split_embeddings[i+1]['Embedding'] = np.add(split_embeddings[i]['Embedding'], split_embeddings[i+1]['Embedding'])
                to_remove.append(i)
            else:
                split_embeddings[i-1]['Text'] += split_embeddings[i]['Text']
                split_embeddings[i-1]['Embedding'] = np.add(split_embeddings[i]['Embedding'], split_embeddings[i-1]['Embedding'])
                to_remove.append(i)
    split_embeddings = {k: v for k, v in split_embeddings.items() if k not in to_remove}
    return split_embeddings

def find_max(split_embeddings):
    return max(d['Score'] for d in split_embeddings.values())

def find_split(embed_model, text1, text2='', min_len = 200, max_len=2000, overlap=0):
    text = text1 + text2
    splits = []
    n_splits = len(text) / min_len
    n_splits = np.random.randint(2,n_splits)
    part_size = len(text) // n_splits
    splits = [text[i:i+part_size+overlap] for i in range(0, len(text), part_size)]
        
    embeddings = []
    for split in splits:
        embeddings.append(np.array(embed_model.create_embedding(split)['data'][0]['embedding']).reshape(1, -1))
    scores = []
    for i in range(len(embeddings) - 1):
        score = cosine_similarity(embeddings[i], embeddings[i+1])[0][0]
        scores.append((sum(scores) + score) / (len(scores) + 1))
    try:
        split_index = scores.index(min(scores))
        if split_index != 0:
            text1 = ' '.join(splits[:split_index])
            text2 = ' '.join(splits[split_index:])
        else:
            text1 = ' '.join(splits[:split_index+1])
            text2 = ' '.join(splits[split_index+1:])
    except:
        print(scores)
        print(len(splits))
        print(len(embeddings))
        print(split_index)
    response = []
    if(len(text1) < max_len):
        response.extend([text1])
    elif len(text1) > max_len:
        text1 = find_split(embed_model, text1=text1, min_len=min_len, max_len=max_len, overlap=overlap)
        response.extend(text1)
    if(len(text2) < max_len):
        response.extend([text2])
    elif len(text2) > max_len:
        text2 = find_split(embed_model, text1=text2, min_len=min_len, max_len=max_len, overlap=overlap)
        response.extend(text2)
    return list(set(response))

def break_bigger(embed_model, split_embeddings, min_len, max_len, overlap):
    to_add = []
    to_remove = []
    for i in split_embeddings.keys():
        if len(split_embeddings[i]['Text']) > max_len:
            smaller_chunks = find_split(embed_model, split_embeddings[i]['Text'], min_len=min_len, max_len=max_len, overlap=overlap)
            to_add.extend(smaller_chunks)
            to_remove.append(i)
    to_add = list(set(to_add))
    for i in range(len(to_add)):
        text = to_add[i]
        if len(text) > max_len:
            raise ValueError("Length of new chunk is greater than maximum length")
        embed = np.array(embed_model.create_embedding(text)['data'][0]['embedding']).reshape(1, -1)
        split_embeddings[i+len(split_embeddings.keys())] = {'Text' : text, 'Embedding' : embed, 'Score' : 0, 'can_join' : False}
    split_embeddings = {k: v for k, v in split_embeddings.items() if k not in to_remove}
    return split_embeddings

def combine_embeddings(embed_model, max_len, threshold, split_embeddings, min_len, overlap, repeat=False):
    split_embeddings = combine_smaller(split_embeddings, min_len)
    split_embeddings = break_bigger(embed_model, split_embeddings, min_len, max_len, overlap)
    temp_dict = {i: split_embeddings[k] for i, k in enumerate(sorted(split_embeddings.keys()))}
    split_embeddings = temp_dict
    split_embeddings = find_scores(split_embeddings)
    max_score = find_max(split_embeddings)
    counter = 1
    while max_score > threshold:
        to_delete = []
        texts_add = []
        for i in split_embeddings.keys():
            if(split_embeddings[i]['Score'] > threshold and split_embeddings[i]['can_join'] == True):
                if(len(split_embeddings[i]['Text'] + split_embeddings[i+1]['Text'])) < max_len:
                    split_embeddings[i+1]['Embedding'] = np.add(split_embeddings[i]['Embedding'], split_embeddings[i+1]['Embedding'])
                    split_embeddings[i+1]['Text']      = split_embeddings[i]['Text'] + split_embeddings[i+1]['Text']
                    to_delete.append(i)
                else:
                    smaller_chunks = find_split(embed_model, split_embeddings[i]['Text'], split_embeddings[i+1]['Text'], min_len=min_len, max_len=max_len, overlap=overlap)
                    texts_add.append(smaller_chunks)
                    to_delete.extend([i, i+1])
        to_delete = set(to_delete)
        num_chunks = list(split_embeddings.keys())[-1]
        new_chunks = 1
        for j in range(len(texts_add)):
            for i in range(len(texts_add[j])):
                text = texts_add[j][i]
                embed = np.array(embed_model.create_embedding(text)['data'][0]['embedding']).reshape(1, -1)
                split_embeddings[num_chunks + new_chunks] = {'Text' : text, 'Embedding' : embed, 'Score' : 0}
                if i == len(texts_add[j]) - 1:
                    split_embeddings[num_chunks + new_chunks]['can_join'] = True
                else:
                    split_embeddings[num_chunks + new_chunks]['can_join'] = False
                new_chunks += 1
        if(len(to_delete) == 0) and repeat:
            return split_embeddings
        elif len(to_delete) == 0:
            repeat = True
            continue
        split_embeddings = {k: v for k, v in split_embeddings.items() if k not in to_delete}
        final_len = len(split_embeddings.keys())
        temp_dict = {i: split_embeddings[k] for i, k in enumerate(sorted(split_embeddings.keys()))}
        split_embeddings = temp_dict
        find_scores(split_embeddings)
        max_score = find_max(split_embeddings)
        print(f"After {counter} iterations, the number of splits are : {final_len}. The highest similarity score is : {max_score}")
        counter += 1
        repeat = False
    return split_embeddings