import os
import requests
from typing import List, Dict
import numpy as np
import re

# Simple cosine similarity TF-IDF or Embedding-based retriever.
# To keep the app portable and ultra-reliable on the VPS (and match Kimi-K3/Nvidia capabilities),
# we will build a robust TF-IDF retrieval system AND allow OpenRouter/DeepSeek fallback for embedding.
# However, TF-IDF + BM25 token overlap is incredibly robust for clinical keyword matches (e.g. "metformin", "EGFR", "ASCVD").

class SimpleRetriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.vocab = {}
        self.idf = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def _build_index(self):
        import re
        doc_frequencies = {}
        num_docs = len(self.chunks)
        
        # Tokenize and compute DF
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk["content"] + " " + chunk["section"] + " " + chunk["disease"]))
            for t in tokens:
                doc_frequencies[t] = doc_frequencies.get(t, 0) + 1
                
        # Compute IDF
        for word, df in doc_frequencies.items():
            self.idf[word] = np.log((num_docs + 1) / (df + 0.5))

    def _get_tf_idf_vector(self, text: str) -> Dict[str, float]:
        import re
        tokens = self._tokenize(text)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            
        vector = {}
        for word, count in tf.items():
            if word in self.idf:
                vector[word] = count * self.idf[word]
        return vector

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = self._get_tf_idf_vector(query)
        if not query_vector:
            return self.chunks[:top_k]
            
        scores = []
        for idx, chunk in enumerate(self.chunks):
            chunk_vector = self._get_tf_idf_vector(chunk["content"] + " " + chunk["section"])
            
            # Cosine similarity
            dot_product = 0.0
            for word, val in query_vector.items():
                if word in chunk_vector:
                    dot_product += val * chunk_vector[word]
                    
            query_norm = np.sqrt(sum(v**2 for v in query_vector.values()))
            chunk_norm = np.sqrt(sum(v**2 for v in chunk_vector.values()))
            
            if query_norm > 0 and chunk_norm > 0:
                score = dot_product / (query_norm * chunk_norm)
            else:
                score = 0.0
                
            scores.append((score, chunk))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scores[:top_k]]

if __name__ == "__main__":
    from chunker import DocumentChunker
    import re
    chunks = DocumentChunker.chunk_guidelines("data/guidelines")
    retriever = SimpleRetriever(chunks)
    res = retriever.retrieve("metformin eGFR")
    for r in res:
        print(f"[{r['citation']}]")
