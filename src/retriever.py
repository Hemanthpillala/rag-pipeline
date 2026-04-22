"""
Retriever: loads FAISS index and retrieves top-k relevant chunks.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List


class FAISSRetriever:
    def __init__(self, index_path: str, k: int = 4):
        self.k = k
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        return self.vectorstore.similarity_search_with_score(query, k=self.k)
