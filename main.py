import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# OpenAI SDK (pip install openai)
from openai import OpenAI

APP_TITLE = "Semantic Search + Rerank (News)"
DEFAULT_DOCS_PATH = os.getenv("DOCS_PATH", "data/news.jsonl")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4.1-mini")  # can change

MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "6000"))  # truncate long docs for speed/cost

client = OpenAI()  # uses OPENAI_API_KEY env var

app = FastAPI(title=APP_TITLE)

# In-memory stores
DOCS: List[Dict[str, Any]] = []
DOC_EMB: Optional[np.ndarray] = None  # shape (N, D), unit-normalized


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(12, ge=1, le=50)
    rerank: bool = True
    rerankK: int = Field(7, ge=1, le=50)


class SearchResult(BaseModel):
    id: Any
    score: float
    content: str
    metadata: Dict[str, Any] = {}


def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Docs file not found at {path}. Set DOCS_PATH env var or create data/news.jsonl"
        )
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}")
            if "content" not in obj:
                raise ValueError(f"Missing 'content' on line {line_no}")
            if "id" not in obj:
                obj["id"] = line_no - 1
            if "metadata" not in obj or obj["metadata"] is None:
                obj["metadata"] = {}
            # truncate for embedding + rerank cost control
            obj["content"] = str(obj["content"])[:MAX_DOC_CHARS]
            docs.append(obj)
    return docs


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings API: returns list of vectors
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs


def _cache_paths():
    return (
        os.path.join(CACHE_DIR, "doc_emb.npy"),
        os.path.join(CACHE_DIR, "doc_meta.json"),
    )


def load_or_build_index():
    global DOCS, DOC_EMB
    _ensure_dirs()
    emb_path, meta_path = _cache_paths()

    # Load docs fresh (so content/metadata matches file)
    DOCS = _read_jsonl(DEFAULT_DOCS_PATH)

    # If cache matches doc count, load embeddings
    if os.path.exists(emb_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cached_n = int(meta.get("n_docs", -1))
            cached_model = meta.get("embed_model", "")
            if cached_n == len(DOCS) and cached_model == EMBED_MODEL:
                DOC_EMB = np.load(emb_path)
                return
        except Exception:
            pass  # fall through and rebuild

    # Build embeddings (batch for speed)
    texts = [d["content"] for d in DOCS]
    batch_size = 64
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        vecs = _embed_texts(chunk)
        all_vecs.append(vecs)
    DOC_EMB = _unit_normalize(np.vstack(all_vecs))

    np.save(emb_path, DOC_EMB)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"n_docs": len(DOCS), "embed_model": EMBED_MODEL}, f)


def _cosine_topk(query_vec: np.ndarray, k: int):
    # DOC_EMB is unit-normalized; query_vec must be unit-normalized
    sims = DOC_EMB @ query_vec  # (N,)
    # top-k indices
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]  # sort desc
    return idx, sims[idx]


def _to_0_1_cosine(sim: float) -> float:
    # cosine is [-1,1] => map to [0,1]
    v = (float(sim) + 1.0) / 2.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _rerank_llm(query: str, candidates: List[Dict[str, Any]]) -> Dict[Any, float]:
    """
    Returns dict {doc_id: score_0_to_1}.
    Uses structured JSON output to avoid parsing issues.
    """
    # Keep prompt compact to control cost/latency
    items = [{"id": c["id"], "content": c["content"][:MAX_DOC_CHARS]} for c in candidates]

    schema = {
        "name": "rerank_scores",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {},
                            "score": {"type": "number", "minimum": 0, "maximum": 10},
                        },
                        "required": ["id", "score"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    prompt = (
        "You are a re-ranking model for semantic search.\n"
        f"Query: {query}\n\n"
        "Rate each document's relevance to the query on a 0-10 scale.\n"
        "10 = extremely relevant, 0 = not relevant.\n"
        "Return JSON matching the schema.\n\n"
        f"Documents: {json.dumps(items, ensure_ascii=False)}"
    )

    # Chat Completions with JSON schema (Structured Outputs)
    try:
        resp = client.chat.completions.create(
            model=RERANK_MODEL,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": schema},
            messages=[
                {"role": "system", "content": "Return only valid JSON that matches the schema."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        out = {}
        for row in data["scores"]:
            out[row["id"]] = max(0.0, min(1.0, float(row["score"]) / 10.0))
        return out
    except Exception as e:
        # Fail gracefully: if rerank fails, we’ll just skip rerank
        raise RuntimeError(f"Rerank failed: {e}") from e


@app.on_event("startup")
def startup_event():
    load_or_build_index()


@app.get("/health")
def health():
    return {"ok": True, "totalDocs": len(DOCS), "embedModel": EMBED_MODEL, "rerankModel": RERANK_MODEL}


@app.post("/search")
def search(req: SearchRequest):
    global DOC_EMB
    t0 = time.perf_counter()

    if DOC_EMB is None or not DOCS:
        raise HTTPException(status_code=500, detail="Index not loaded")

    q = req.query.strip()
    if not q:
        return {"results": [], "reranked": False, "metrics": {"latency": 0, "totalDocs": len(DOCS)}}

    # Stage 1: vector retrieval
    qvec = _unit_normalize(_embed_texts([q]))[0]
    idx, sims = _cosine_topk(qvec, req.k)

    candidates = []
    for i, s in zip(idx.tolist(), sims.tolist()):
        doc = DOCS[i]
        candidates.append(
            {
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "base_score": _to_0_1_cosine(s),
            }
        )

    reranked = False
    # Stage 2: rerank
    if req.rerank and candidates:
        try:
            score_map = _rerank_llm(q, candidates)
            for c in candidates:
                if c["id"] in score_map:
                    c["score"] = float(score_map[c["id"]])
                else:
                    c["score"] = c["base_score"]
            reranked = True
        except Exception:
            # If rerank fails, fall back to base scores
            for c in candidates:
                c["score"] = c["base_score"]
            reranked = False
    else:
        for c in candidates:
            c["score"] = c["base_score"]

    # Sort by final score desc, stable tie-break by id
    candidates.sort(key=lambda x: (-x["score"], str(x["id"])))

    top = candidates[: req.rerankK if req.rerank else req.k]
    results = [
        {
            "id": r["id"],
            "score": float(max(0.0, min(1.0, r["score"]))),
            "content": r["content"],
            "metadata": {"source": r["metadata"].get("source", "unknown"), **r["metadata"]},
        }
        for r in top
    ]

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"results": results, "reranked": reranked, "metrics": {"latency": latency_ms, "totalDocs": len(DOCS)}}
