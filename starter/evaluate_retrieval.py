"""Chương trình triển khai và so sánh 3 kỹ thuật Retrieval cốt lõi của Day 7:
1. Kỹ thuật 1: Dense Retrieval (Vector Store ChromaDB Day 6 + all-MiniLM-L6-v2)
2. Kỹ thuật 2: Lexical Retrieval (Okapi BM25 k1=1.5, b=0.75)
3. Kỹ thuật 3: Hybrid Retrieval (Dense + BM25 qua Reciprocal Rank Fusion - RRF)
Cùng 2 cấu hình cải thiện theo chuẩn rubric:
- Cải thiện 1: Tối ưu Context Budget (budget=1200)
- Cải thiện 2: Metadata Filtering (lọc theo chapter)
Và đánh giá cấu hình chốt trên Test độc lập.
"""
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# Cấu hình offline cho SentenceTransformer
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "starter"))
from baseline import hit, pack_context


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class BM25Index:
    """Hiện thực chuẩn thuật toán Okapi BM25 với chuẩn hóa độ dài văn bản (k1=1.5, b=0.75)."""
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_lens = [len(tokenize(c['text'])) for c in chunks]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_freqs = []
        self.nd = {}
        for c in chunks:
            counts = Counter(tokenize(c['text']))
            self.doc_freqs.append(counts)
            for word in counts:
                self.nd[word] = self.nd.get(word, 0) + 1
        self.idf = {}
        for word, freq in self.nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens, doc_idx):
        doc_len = self.doc_lens[doc_idx]
        counts = self.doc_freqs[doc_idx]
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0)
        s = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            tf = counts[token]
            token_idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * len_norm
            s += token_idf * (numerator / denominator)
        return s

    def search(self, query, top_k=3, filter_dict=None):
        query_tokens = tokenize(query)
        scores = []
        for idx, chunk in enumerate(self.chunks):
            if filter_dict:
                meta = chunk.get("metadata", {})
                match = all(meta.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            sc = self.score(query_tokens, idx)
            scores.append((chunk["id"], sc))
        scores.sort(key=lambda x: -x[1])
        return [{"id": cid, "score": sc} for cid, sc in scores[:top_k]]


def run_dense_search(collection, model, query, top_k=3, filter_dict=None):
    """Truy vấn tìm kiếm Dense trên ChromaDB bằng sentence-transformers."""
    q_emb = model.encode(query).tolist()
    where = None
    if filter_dict and "chapter" in filter_dict:
        where = {"chapter": filter_dict["chapter"]}
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, where=where)
    retrieved_ids = res["ids"][0] if res and "ids" in res and res["ids"] else []
    return retrieved_ids, res


def evaluate_configuration(name, search_fn, questions, chunks_by_id, budget=800, split="dev"):
    """Đánh giá một cấu hình retrieval: đo Hit@1, Evidence Hit@3 và Context Evidence Rate."""
    split_questions = [q for q in questions if q.get("split") == split]
    if not split_questions:
        return {}

    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    total = len(split_questions)
    traces = []
    start_t = time.perf_counter()

    for q in split_questions:
        retrieved_ids = search_fn(q)
        retrieved_chunks = [chunks_by_id[cid] for cid in retrieved_ids if cid in chunks_by_id]
        
        # Check hits
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in retrieved_chunks]
        top1_hit = bool(marked and marked[0]['hit'])
        top3_hit = any(c['hit'] for c in marked[:3])
        
        ctx_text, ctx_ids = pack_context(retrieved_chunks[:3], budget=budget)
        ctx_hit = any(c['hit'] and c['id'] in ctx_ids for c in marked[:3])

        if top1_hit:
            hit1_count += 1
        if top3_hit:
            hit3_count += 1
        if ctx_hit:
            context_hit_count += 1

        traces.append({
            "id": q["id"],
            "question": q["question"],
            "retrieved_ids": retrieved_ids[:3],
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "context_hit": ctx_hit,
            "context_chars": len(ctx_text),
        })

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / total
    return {
        "config": name,
        "split": split,
        "total": total,
        "hit1": f"{hit1_count}/{total} ({hit1_count/total*100:.1f}%)",
        "hit3": f"{hit3_count}/{total} ({hit3_count/total*100:.1f}%)",
        "context_rate": f"{context_hit_count}/{total} ({context_hit_count/total*100:.1f}%)",
        "latency_ms": f"{elapsed_ms:.2f}ms",
        "traces": traces,
    }


def main():
    print("=" * 80)
    print("BƯỚC 2: TRIỂN KHAI & ĐÁNH GIÁ 3 KỸ THUẬT RETRIEVAL DAY 07")
    print("=" * 80)

    # 1. Đọc dữ liệu
    chunks_path = ROOT_DIR / "data" / "chunks.json"
    questions_path = ROOT_DIR / "data" / "questions.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_by_id = {c["id"]: c for c in chunks}

    # 2. Khởi tạo Index
    print("-> Đang khởi tạo Kỹ thuật 2: Lexical BM25 Index (k1=1.5, b=0.75)...")
    bm25_index = BM25Index(chunks, k1=1.5, b=0.75)

    print("-> Đang kết nối Kỹ thuật 1: ChromaDB và load model all-MiniLM-L6-v2 (offline)...")
    day6_chroma_dir = ROOT_DIR / "starter" / "data" / "chroma_db"
    import chromadb
    from sentence_transformers import SentenceTransformer

    chroma_client = chromadb.PersistentClient(path=str(day6_chroma_dir))
    col = chroma_client.get_collection("ingestion_strategy_b")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    results = []

    # --- Kỹ thuật 1: Dense Retrieval (ChromaDB Day 6, budget=800) trên Dev ---
    def search_dense(q):
        ids, _ = run_dense_search(col, dense_model, q["question"], top_k=3)
        return ids
    results.append(evaluate_configuration("1. Dense Retrieval (ChromaDB Day 6)", search_dense, questions, chunks_by_id, budget=800, split="dev"))

    # --- Kỹ thuật 2: Lexical Retrieval (Okapi BM25, budget=800) trên Dev ---
    def search_bm25_base(q):
        res = bm25_index.search(q["question"], top_k=3)
        return [r["id"] for r in res]
    results.append(evaluate_configuration("2. Lexical Retrieval (Okapi BM25)", search_bm25_base, questions, chunks_by_id, budget=800, split="dev"))

    # --- Kỹ thuật 3: Hybrid Retrieval (Dense + BM25 RRF, budget=800) trên Dev ---
    def search_hybrid(q):
        bm25_candidates = [r["id"] for r in bm25_index.search(q["question"], top_k=10)]
        dense_candidates, _ = run_dense_search(col, dense_model, q["question"], top_k=10)
        
        rrf = {}
        for rank, cid in enumerate(bm25_candidates):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        for rank, cid in enumerate(dense_candidates):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            
        ranked = sorted(rrf.keys(), key=lambda cid: -rrf[cid])[:3]
        return ranked
    results.append(evaluate_configuration("3. Hybrid Retrieval (Dense + BM25 RRF)", search_hybrid, questions, chunks_by_id, budget=800, split="dev"))

    # --- Cải thiện 1: Tối ưu Context Budget (BM25, budget=1200) trên Dev ---
    results.append(evaluate_configuration("4. Cải thiện 1: Tối ưu Budget (BM25, b=1200)", search_bm25_base, questions, chunks_by_id, budget=1200, split="dev"))

    # --- Cải thiện 2: BM25 + Metadata Filter (budget=1200) trên Dev ---
    def search_bm25_filtered(q):
        res = bm25_index.search(q["question"], top_k=10)
        filt = q.get("filter")
        if filt and "chapter" in filt:
            target_chap = filt["chapter"]
            filtered_res = [r for r in res if chunks_by_id[r["id"]].get("metadata", {}).get("chapter") == target_chap]
            if filtered_res:
                return [r["id"] for r in filtered_res[:3]]
        return [r["id"] for r in res[:3]]
    results.append(evaluate_configuration("5. Cải thiện 2: BM25 + Filter (k=3, b=1200)", search_bm25_filtered, questions, chunks_by_id, budget=1200, split="dev"))

    # --- Cấu hình chốt trên tập Test độc lập ---
    results.append(evaluate_configuration("6. Cấu hình chốt trên Test (BM25+Filter, b=1200)", search_bm25_filtered, questions, chunks_by_id, budget=1200, split="test"))

    # 4. Xuất bảng kết quả
    print("\n" + "=" * 105)
    print(f"{'Kỹ thuật / Cấu hình':<48} | {'Split':<5} | {'Hit@1':<14} | {'Evidence Hit@3':<14} | {'Context Rate':<14}")
    print("-" * 105)
    for r in results:
        print(f"{r['config']:<48} | {r['split']:<5} | {r['hit1']:<14} | {r['hit3']:<14} | {r['context_rate']:<14}")
    print("=" * 105)

    # 5. Lưu kết quả ra file JSON
    out_dir = ROOT_DIR / "starter" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_save = [{k: v for k, v in r.items() if k != "traces"} for r in results]
    with open(out_dir / "retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_save, f, ensure_ascii=False, indent=2)
    print(f"\n[XONG] Kết quả chi tiết đã được lưu tại: {out_dir / 'retrieval_comparison.json'}")


if __name__ == "__main__":
    main()
