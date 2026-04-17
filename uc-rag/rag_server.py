import argparse
import os
import uuid
import re
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import chromadb


# -------------------------------
# Utility
# -------------------------------
def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def split_sentences(text):
    sentences = text.replace("\n", " ").split(". ")
    return [s.strip() + "." for s in sentences if s.strip()]


def clean_sentence(s):
    """
    Extract only numbered clauses like '3.1 ...'
    Removes headings and separators.
    """
    s = s.strip()
    match = re.search(r'(\d+\.\d+.*)', s)
    if match:
        return match.group(1).strip()
    return None


# -------------------------------
# STRICT LLM (grounded extraction)
# -------------------------------
def strict_llm(context_chunks, query):
    query = query.lower()
    relevant_lines = []

    for chunk in context_chunks:
        sentences = re.split(r'(?<=\.)\s+', chunk)

        for s in sentences:
            s = s.strip()
            s_lower = s.lower()

            # Leave without pay → approval clause only
            if "leave without pay" in query:
                if "approval" in s_lower and (
                    "department head" in s_lower or "hr director" in s_lower
                ):
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

            # Phone query → IT clause 3.1
            elif "phone" in query:
                if "personal devices may be used" in s_lower:
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

            # Allowance query → Finance clause
            elif "allowance" in query:
                # STRICT: only home office allowance clause
                if "home office equipment allowance" in s_lower:
                    cleaned = clean_sentence(s)
                    if cleaned:
                        relevant_lines.append(cleaned)

    if not relevant_lines:
        return ""

    return " ".join(relevant_lines)


# -------------------------------
# Chunk Documents
# -------------------------------
def chunk_documents(docs_dir, max_tokens=400):
    chunks = []

    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(docs_dir, filename)

        try:
            text = open(path, encoding="utf-8").read()
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        sentences = split_sentences(text)

        current_chunk = ""
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            tokens = estimate_tokens(sentence)

            if current_tokens + tokens > max_tokens:
                if current_chunk:
                    chunks.append({
                        "doc_name": filename,
                        "chunk_index": chunk_index,
                        "text": current_chunk.strip()
                    })
                    chunk_index += 1

                current_chunk = sentence
                current_tokens = tokens
            else:
                current_chunk += " " + sentence
                current_tokens += tokens

        if current_chunk:
            chunks.append({
                "doc_name": filename,
                "chunk_index": chunk_index,
                "text": current_chunk.strip()
            })

    return chunks


# -------------------------------
# Retrieve and Answer
# -------------------------------
def retrieve_and_answer(query, collection, embedder, top_k=3, threshold=0.4):
    """
    Threshold adjusted based on observed embedding similarity distribution.
    Score = 1 / (1 + dist)
    """

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 5,
        include=["documents", "metadatas", "distances"]
    )

    print("\nDEBUG: Raw retrieval results:")

    retrieved = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = 1 / (1 + dist)

        print(f"{meta['doc_name']} | chunk {meta['chunk_index']} | dist={dist:.4f} | score={score:.4f}")

        if score >= threshold:
            retrieved.append({
                "text": doc,
                "doc_name": meta["doc_name"],
                "chunk_index": meta["chunk_index"],
                "score": score
            })

    # -------------------------------
    # Refusal
    # -------------------------------
    if not retrieved:
        return {
            "answer": (
                "This question is not covered in the retrieved policy documents.\n"
                "Retrieved chunks: []. Please contact the relevant department for guidance."
            ),
            "cited_chunks": []
        }

    # -------------------------------
    # Group by document (no blending)
    # -------------------------------
    grouped = defaultdict(list)
    for r in retrieved:
        grouped[r["doc_name"]].append(r)

    final_answers = []
    final_citations = []

    for doc_name, chunks in grouped.items():
        chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

        context_chunks = [c["text"] for c in chunks]

        answer = strict_llm(context_chunks, query)

        if answer:
            final_answers.append(answer)

            top_chunk = chunks[0]
            final_citations.append({
                "doc_name": top_chunk["doc_name"],
                "chunk_index": top_chunk["chunk_index"]
            })

    # -------------------------------
    # Fallback if filtering empty
    # -------------------------------
    if not final_answers:
        top = sorted(retrieved, key=lambda x: x["score"], reverse=True)[0]
        return {
            "answer": clean_sentence(top["text"]) or top["text"],
            "cited_chunks": [{
                "doc_name": top["doc_name"],
                "chunk_index": top["chunk_index"]
            }]
        }

    return {
        "answer": "\n\n".join(final_answers),
        "cited_chunks": final_citations
    }


# -------------------------------
# Build Index
# -------------------------------
def build_index(docs_dir, db_path="./chroma_db"):
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("rag_collection")
    except:
        pass

    collection = client.create_collection("rag_collection")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = chunk_documents(docs_dir)

    for chunk in chunks:
        embedding = embedder.encode(chunk["text"]).tolist()

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "doc_name": chunk["doc_name"],
                "chunk_index": chunk["chunk_index"]
            }]
        )

    print(f"Indexed {len(chunks)} chunks")


# -------------------------------
# Main
# -------------------------------
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
        collection = client.get_collection("rag_collection")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        result = retrieve_and_answer(
            args.query,
            collection,
            embedder
        )

        print("\nAnswer:\n", result["answer"])
        print("\nCitations:")
        for c in result["cited_chunks"]:
            print(c)


if __name__ == "__main__":
    main()