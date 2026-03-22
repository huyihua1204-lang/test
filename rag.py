"""
RAG 检索增强模块
管理 ChromaDB 向量知识库，支持文档入库、检索与删除。
"""

import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DB_PATH       = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION    = "knowledge_base"
EMBED_MODEL   = os.getenv("EMBED_MODEL_PATH", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 60


def _collection():
    """获取（或创建）持久化向量集合。"""
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk(text: str) -> list[str]:
    """将文本按固定窗口切块，相邻块有重叠以保留上下文。"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


# ── 公开 API ──────────────────────────────────────────────────────────────────

def health_check() -> dict:
    """
    检查向量库是否可用。
    返回值：{"ok": bool, "count": int, "error": str}
    """
    try:
        col = _collection()
        return {"ok": True, "count": col.count(), "error": ""}
    except Exception as e:
        return {"ok": False, "count": 0, "error": str(e)}

def add_document(filename: str, text: str) -> int:
    """
    将文档切块、向量化并写入知识库（已存在则覆盖）。

    返回值：写入的片段数量。
    """
    col = _collection()
    chunks = _chunk(text)
    if not chunks:
        return 0

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{filename}|{i}|{chunk[:40]}".encode()).hexdigest()
        ids.append(uid)
        docs.append(chunk)
        metas.append({"source": filename, "chunk_index": i})

    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(chunks)


def delete_document(filename: str) -> int:
    """
    删除指定文档的全部片段。

    返回值：删除的片段数量。
    """
    col = _collection()
    hits = col.get(where={"source": filename})
    if hits["ids"]:
        col.delete(ids=hits["ids"])
    return len(hits["ids"])


def list_documents() -> list[dict]:
    """
    返回知识库中所有文档的摘要。

    返回值：[{"filename": str, "chunks": int}, ...]
    """
    col = _collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    tally: dict[str, int] = {}
    for m in all_meta:
        src = m["source"]
        tally[src] = tally.get(src, 0) + 1
    return [{"filename": k, "chunks": v} for k, v in sorted(tally.items())]


def retrieve(query: str, k: int = 3, min_score: float = 0.45) -> list[dict]:
    """
    检索与 query 最相关的 k 个文本片段。

    参数：
        min_score : 相似度阈值（0~1），低于此值的片段被丢弃，避免无关内容干扰 AI。

    返回值：[{"content": str, "source": str, "score": float}, ...]
    """
    col = _collection()
    total = col.count()
    if total == 0:
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    chunks = [
        {
            "content": doc,
            "source": meta["source"],
            "score": round(1 - dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    # 过滤低相关度片段，防止无关内容误导 AI
    return [c for c in chunks if c["score"] >= min_score]
