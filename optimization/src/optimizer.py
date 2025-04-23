from llama_cpp import Llama
import numpy as np
from .utils import *

class ChunkOptimizer:
    def __init__(self, embed_model_path : str = None, embed_model : Llama = None):
        self.model_path = embed_model_path
        self.embed_model = None
        if embed_model:
            self.embed_model = embed_model
        else:
            self.embed_model = Llama(model_path='C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf', 
                                    embedding=True,
                                    verbose=False)
        
    def optimize_chunks(self, max_chunk_len : int, min_chunk_len : int, similarity_threshold : float, overlap=0, base_embeddings = None, embeddings : list = None, chunks : list = None):
        split_embeddings = {}
        if max_chunk_len < (3 * min_chunk_len):
            raise ValueError("Maximum chunk length should be at least 3 times minimum chunk length")
        if base_embeddings is not None:
            split_embeddings = base_embeddings
        else:
            if embeddings:
                if(len(chunks) != len(embeddings)):
                    raise ValueError("The length of chunks and embeddings list should be the same.")
                for i in range(len(chunks)):
                    split_embeddings[i] = {'Text' : chunks[i], 'Embedding' : embeddings[i], 'Score' : 0}
            else:
                for i in range(len(chunks)):
                    text = chunks[i]
                    embedding = np.array(self.embed_model.create_embedding(text)['data'][0]['embedding']).reshape(1, -1)
                    split_embeddings[i] = {'Text' : text, 'Embedding' : embedding, 'Score' : 0}
            find_scores(split_embeddings)
        split_embeddings = combine_embeddings(self.embed_model, max_chunk_len, similarity_threshold, split_embeddings, min_chunk_len, overlap=overlap)
        return split_embeddings
            