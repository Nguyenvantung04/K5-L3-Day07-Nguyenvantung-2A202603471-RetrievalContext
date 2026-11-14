"""Script test nhanh một câu hỏi bất kỳ qua cả 3 phương pháp Retrieval:
1. Lexical BM25 (Okapi BM25, k1=1.5, b=0.75)
2. Dense ChromaDB (all-MiniLM-L6-v2)
3. Hybrid RRF (kết hợp cả hai)
"""
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "starter"))
from evaluate_retrieval import BM25Index

def main():
    parser = argparse.ArgumentParser(description="Test thử 3 phương pháp retrieval với 1 câu hỏi")
    parser.add_argument("--query", type=str, default="Thời gian học tập tối đa cho chương trình đào tạo chuẩn 4 năm là bao lâu?",
                        help="Câu hỏi cần tìm kiếm")
    parser.add_argument("--top_k", type=int, default=3, help="Số lượng kết quả lấy ra")
    args = parser.parse_args()

    query = args.query
    top_k = args.top_k

    print("\n" + "=" * 80)
    print(f"CÂU HỎI TEST: \"{query}\"")
    print("=" * 80)

    # 1. Đọc Chunks
    chunks_path = ROOT_DIR / "data" / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_by_id = {c["id"]: c for c in chunks}

    # 2. Phương pháp 1: Okapi BM25
    bm25 = BM25Index(chunks)
    bm25_res = bm25.search(query, top_k=top_k)
    print("\n[1] KẾT QUẢ LEXICAL RETRIEVAL (Okapi BM25):")
    for rank, item in enumerate(bm25_res, 1):
        print(f"   Rank {rank}: [{item['id']} | Trang {item['page']}] - Điểm BM25: {item['score']:.4f}")
        first_line = item['text'].split('\n')[0][:90]
        print(f"           Trích đoạn: {first_line}...")

    # 3. Phương pháp 2: Dense ChromaDB
    import chromadb
    from sentence_transformers import SentenceTransformer
    local_chroma = ROOT_DIR / "starter" / "data" / "chroma_db"
    chroma_path = local_chroma if local_chroma.exists() else ROOT_DIR.parent / "K5-L3A-Day06-NguyenVanTung-2A202603471-DataIngestionEngineering" / "starter" / "data" / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection("ingestion_strategy_b")
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    vec = model.encode(query).tolist()
    dense_res = col.query(query_embeddings=[vec], n_results=top_k)
    dense_ids = dense_res["ids"][0]
    dense_dists = dense_res["distances"][0]

    print("\n[2] KẾT QUẢ DENSE RETRIEVAL (ChromaDB + all-MiniLM-L6-v2):")
    for rank, (cid, dist) in enumerate(zip(dense_ids, dense_dists), 1):
        c = chunk_by_id[cid]
        sim = 1.0 - dist
        print(f"   Rank {rank}: [{cid} | Trang {c['page']}] - Cosine Sim: {sim:.4f}")
        first_line = c['text'].split('\n')[0][:90]
        print(f"           Trích đoạn: {first_line}...")

    # 4. Phương pháp 3: Hybrid RRF
    bm25_cands = [r["id"] for r in bm25.search(query, top_k=10)]
    dense_cands = col.query(query_embeddings=[vec], n_results=10)["ids"][0]

    rrf = {}
    for rank, cid in enumerate(bm25_cands):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
    for rank, cid in enumerate(dense_cands):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)

    hybrid_ids = sorted(rrf.keys(), key=lambda cid: -rrf[cid])[:top_k]

    print("\n[3] KẾT QUẢ HYBRID RETRIEVAL (Dense + BM25 qua RRF):")
    for rank, cid in enumerate(hybrid_ids, 1):
        c = chunk_by_id[cid]
        score = rrf[cid]
        print(f"   Rank {rank}: [{cid} | Trang {c['page']}] - Điểm RRF: {score:.5f}")
        first_line = c['text'].split('\n')[0][:90]
        print(f"           Trích đoạn: {first_line}...")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()


