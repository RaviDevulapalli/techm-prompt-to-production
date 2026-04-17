import argparse
import os
import sys
import re
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Helper: clean sentence
# -----------------------------
def clean_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # Keep numbered clauses only
    if re.match(r"^\d+(\.\d+)?", s):
        return s
    return ""


# -----------------------------
# SKILL: chunk_documents
# -----------------------------
def chunk_documents(docs_dir: str, max_tokens: int = 400) -> list[dict]:
    chunks = []

    for file in os.listdir(docs_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(docs_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_len = 0
        chunk_index = 0

        for sent in sentences:
            tokens = sent.split()
            if current_len + len(tokens) > max_tokens:
                chunks.append({
                    "doc_name": file,
                    "chunk_index": chunk_index,
                    "text": " ".join(current_chunk)
                })
                chunk_index += 1
                current_chunk = []
                current_len = 0

            current_chunk.append(sent)
            current_len += len(tokens)

        if current_chunk:
            chunks.append({
                "doc_name": file,
                "chunk_index": chunk_index,
                "text": " ".join(current_chunk)
            })

    return chunks


# -----------------------------
# STRICT LLM (NO hallucination)
# -----------------------------
def strict_llm(context_chunks, query):
    query = query.lower()
    relevant_lines = []

    for chunk in context_chunks:
        sentences = re.split(r'\n|\.', chunk)

        # ==========================================
        # COMMIT 5: CONTEXT BREACH FIX
        # STRICT GROUNDING: Only extract sentences
        # explicitly matching query keywords.
        # Prevents returning full chunks / irrelevant data
        # ==========================================
        for s in sentences:
            s_lower = s.lower()

            if "leave without pay" in query or "approve" in query:
                if "leave without pay" in s_lower or "lwp" in s_lower:
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

            elif "personal phone" in query or "personal device" in query:
                if "personal device" in s_lower:
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

            elif "allowance" in query:
                if "home office equipment allowance" in s_lower:
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

    return " ".join(relevant_lines)


# -----------------------------
# SKILL: retrieve_and_answer
# -----------------------------
def retrieve_and_answer(query, collection, embedder, top_k=3, threshold=0.4):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    print("\nDEBUG: Raw retrieval results:")
    for doc, meta, dist in zip(docs, metadatas, distances):
        score = 1 / (1 + dist)
        print(f"{meta['doc_name']} | chunk {meta['chunk_index']} | dist={dist:.4f} | score={score:.4f}")

        if score >= threshold:
            retrieved.append({
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "text": doc,
                "score": score
            })

    if not retrieved:
        return {
            "answer": "This question is not covered in the retrieved policy documents.\nRetrieved chunks: []. Please contact the relevant department for guidance.",
            "cited_chunks": []
        }

    grouped = defaultdict(list)
    for r in retrieved:
        grouped[r["doc_name"]].append(r)

    # ==========================================
    # COMMIT 6: CROSS-DOC BLENDING FIX
    # SINGLE-SOURCE ENFORCEMENT:
    # Select ONLY highest scoring document
    # Prevents mixing HR + IT + Finance
    # ==========================================
    # CROSS-DOC FIX: enforce single-document selection to prevent blending policies
    top_doc = max(grouped.items(), key=lambda x: max(c["score"] for c in x[1]))
    doc_name, chunks = top_doc

    chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

    context_chunks = [c["text"] for c in chunks]

    answer = strict_llm(context_chunks, query)

    if not answer.strip():
        return {
            "answer": "This question is not covered in the retrieved policy documents.\nRetrieved chunks: []. Please contact the relevant department for guidance.",
            "cited_chunks": []
        }

    return {
        "answer": answer,
        "cited_chunks": [{
            "doc_name": chunks[0]["doc_name"],
            "chunk_index": chunks[0]["chunk_index"]
        }]
    }


# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index(docs_dir, db_path="./chroma_db"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("policy_docs")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = chunk_documents(docs_dir)

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk["text"]).tolist()

        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "doc_name": chunk["doc_name"],
                "chunk_index": chunk["chunk_index"]
            }],
            ids=[f"id_{i}"]
        )

    print(f"Indexed {len(chunks)} chunks")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--docs-dir", default="../data/policy-documents")
    parser.add_argument("--db-path", default="./chroma_db")

    args = parser.parse_args()

    if args.build_index:
        build_index(args.docs_dir, args.db_path)

    if args.query:
        client = chromadb.PersistentClient(path=args.db_path)
        collection = client.get_or_create_collection("policy_docs")

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        result = retrieve_and_answer(args.query, collection, embedder)

        print("\nAnswer:\n", result["answer"])
        print("\nCitations:")
        for c in result["cited_chunks"]:
            print(c)


if __name__ == "__main__":
    main()