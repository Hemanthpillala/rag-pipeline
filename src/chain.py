"""
RAG chain: retrieves context and generates answers using GPT-4.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from retriever import FAISSRetriever


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise and helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say "I don't have enough context to answer."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    return "\n\n".join(f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs))


class RAGChain:
    def __init__(self, index_path: str = "data/faiss_index",
                 model: str = "gpt-4o-mini", k: int = 4, temperature: float = 0.0):
        self.retriever = FAISSRetriever(index_path, k=k)
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = (
            {
                "context": lambda x: format_docs(self.retriever.retrieve(x["question"])),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        docs = self.retriever.retrieve(question)
        context = format_docs(docs)
        answer = self.chain.invoke({"question": question})
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "context_used": context,
        }
