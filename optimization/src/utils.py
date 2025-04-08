import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def find_scores(split_embeddings):
    similarity_scores = []
    for i in range(len(split_embeddings.keys())):
        if i == len(split_embeddings.keys()) - 1:
            score = 0
        else:
            score = cosine_similarity(split_embeddings[i]['Embedding'], split_embeddings[i+1]['Embedding'])[0][0]
        similarity_scores.append(score)
        
        split_embeddings[i]['Score'] = score
    return similarity_scores

def find_max(split_embeddings):
    return max(d['Score'] for d in split_embeddings.values())


def find_split(embed_model, text1, text2='', chars=['.'], min_len = 200, max_len=1200):
    text = text1 + text2
    
    n_splits = len(text) / min_len
    n_splits = np.random.randint(2,n_splits)
    part_size = len(text) // n_splits
    splits = [text[i:i+part_size] for i in range(0, len(text), part_size)]
        
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
    if((len(text1) > min_len) and (len(text1) < max_len)):
        response.extend([text1])
    elif len(text1) > max_len:
        #print("Entering nested splitter", len(text1))
        text1 = find_split(embed_model, text1, chars=chars)
        response.extend(text1)
    if((len(text2) > min_len) and (len(text2) < max_len)):
        response.extend([text2])
    elif len(text2) > max_len:
        #print("Entering nested splitter", len(text2))
        text2 = find_split(embed_model, text2, chars=chars)
        response.extend(text2)
    return list(set(response))

def combine_embeddings(embed_model, max_len, threshold, split_embeddings, min_len, repeat=False):
    max_score = find_max(split_embeddings)
    counter = 1
    while max_score > threshold:
        to_delete = []
        texts_add = []
        for i in split_embeddings.keys():
            if split_embeddings[i]['Score'] > threshold:
                if len(split_embeddings[i]['Text'] + split_embeddings[i+1]['Text']) < max_len:
                    split_embeddings[i+1]['Embedding'] = np.add(split_embeddings[i]['Embedding'], split_embeddings[i+1]['Embedding'])
                    split_embeddings[i+1]['Text']      = split_embeddings[i]['Text'] + ' ' + split_embeddings[i+1]['Text']
                    to_delete.append(i)
                else:
                    smaller_chunks = find_split(embed_model, split_embeddings[i]['Text'], split_embeddings[i+1]['Text'], ['.', ';', ','], min_len, max_len)
                    smaller_chunks = [x for x in smaller_chunks if len(x) > min_len]
                    texts_add.extend(smaller_chunks)
                    to_delete.extend([i, i+1])
        to_delete = set(to_delete)
        for i in range(len(texts_add)):
            if len(texts_add[i]) > min_len:
                text = texts_add[i]
                if len(text) > max_len:
                    raise ValueError("Length of new chunk is greater than maximum length")
                embed = np.array(embed_model.create_embedding(text)['data'][0]['embedding']).reshape(1, -1)
                split_embeddings[i+len(split_embeddings.keys())] = {'Text' : text, 'Embedding' : embed, 'Score' : 0}
        if(len(to_delete) == 0) and repeat:
            return split_embeddings
        elif len(to_delete) == 0:
            repeat = True
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