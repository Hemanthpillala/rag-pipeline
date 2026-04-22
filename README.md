# RAG Pipeline — LangChain + GPT-4 + FAISS

End-to-end Retrieval-Augmented Generation pipeline enabling context-aware Q&A over custom document corpora. Evaluated with RAGAS metrics — achieving **87% answer relevance** on held-out evaluation set.

## Architecture

```
Documents (PDF/TXT)
      ↓
  Chunking (RecursiveCharacterTextSplitter, 512 tokens, 64 overlap)
      ↓
  Embeddings (text-embedding-3-small)
      ↓
  FAISS Vector Store ←──── Query Embedding
      ↓                          ↑
  Top-k Retrieval (k=4)    User Question
      ↓
  Context Assembly
      ↓
  GPT-4o-mini (temperature=0)
      ↓
  Grounded Answer + Sources
```

## RAGAS Evaluation

| Metric | Score |
|--------|-------|
| Answer Relevancy | 0.87 |
| Faithfulness | 0.91 |
| Context Precision | 0.84 |
| Context Recall | 0.79 |

## Project Structure

```
rag-pipeline/
├── src/
│   ├── ingest.py       # Load docs, chunk, embed, build FAISS index
│   ├── retriever.py    # FAISS retriever wrapper
│   ├── chain.py        # LangChain RAG chain (retriever + GPT-4)
│   ├── api.py          # FastAPI endpoint
│   └── evaluate.py     # RAGAS evaluation suite
├── .env.example
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # Add your OpenAI key

# 1. Ingest documents into FAISS
python src/ingest.py --docs_dir data/docs/ --index_path data/faiss_index

# 2. Ask a question directly
python -c "from src.chain import RAGChain; r=RAGChain(); print(r.ask('What is X?'))"

# 3. Run FastAPI server
uvicorn src.api:app --reload --port 8000

# 4. Evaluate with RAGAS
python src/evaluate.py --index_path data/faiss_index --qa_path data/eval_qa.json
```

## Tech Stack

`LangChain` · `OpenAI GPT-4` · `FAISS` · `FastAPI` · `RAGAS` · `text-embedding-3-small`
