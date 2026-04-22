"""
Document ingestion: load PDFs/text files, chunk, embed, and store in FAISS.

Usage:
    python src/ingest.py --docs_dir data/docs/ --index_path data/faiss_index
"""

import argparse
import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(docs_dir: str):
    loaders = [
        DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader),
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Loader warning: {e}")
    return docs


def chunk_documents(docs, chunk_size: int = 512, chunk_overlap: int = 64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(chunks, index_path: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved to {index_path} ({len(chunks)} chunks)")
    return vectorstore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="data/docs/")
    parser.add_argument("--index_path", type=str, default="data/faiss_index")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading documents from {args.docs_dir}...")
    docs = load_documents(args.docs_dir)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks")

    build_index(chunks, args.index_path)


if __name__ == "__main__":
    main()
