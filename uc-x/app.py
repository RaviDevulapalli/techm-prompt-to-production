"""
UC-X app.py — Commit 5 version

Improvement:
- Added strict refusal enforcement
- Prevents answering when match is weak or ambiguous
- Still uses scoring (final fix comes in next commit)
"""

import argparse
import os
import re

POLICY_FILES = {
    "policy_hr_leave.txt": "../data/policy-documents/policy_hr_leave.txt",
    "policy_it_acceptable_use.txt": "../data/policy-documents/policy_it_acceptable_use.txt",
    "policy_finance_reimbursement.txt": "../data/policy-documents/policy_finance_reimbursement.txt",
}

REFUSAL_RESPONSE = """This question is not covered in the available policy documents
(policy_hr_leave.txt, policy_it_acceptable_use.txt, policy_finance_reimbursement.txt).
Please contact [relevant team] for guidance."""


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents():
    docs = {}
    section_pattern = re.compile(r"^(\d+\.\d+)\b")

    for name, path in POLICY_FILES.items():
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        parsed = []
        current_section = None
        buffer = []

        for line in lines:
            line = line.strip()

            if re.match(r"^[=\-]{5,}$", line):
                continue

            match = section_pattern.match(line)

            if match:
                if current_section:
                    parsed.append({
                        "doc_name": name,
                        "section": current_section,
                        "content": " ".join(buffer)
                    })

                current_section = match.group(1)
                buffer = [line]
            else:
                if current_section:
                    buffer.append(line)

        if current_section:
            parsed.append({
                "doc_name": name,
                "section": current_section,
                "content": " ".join(buffer)
            })

        docs[name] = parsed

    return docs


# -----------------------------
# MATCHING (WITH THRESHOLD)
# -----------------------------
def find_best_match(query, docs):
    query_tokens = set(re.findall(r"\w+", query.lower()))

    best_match = None
    best_score = 0

    for doc_name, sections in docs.items():
        for sec in sections:
            content_tokens = set(re.findall(r"\w+", sec["content"].lower()))

            overlap = query_tokens & content_tokens
            score = len(overlap)

            if score >= 2 and score > best_score:
                best_score = score
                best_match = sec

    return best_match, best_score


# -----------------------------
# ANSWER (STRICT REFUSAL)
# -----------------------------
def answer_question(query, docs):
    match, score = find_best_match(query, docs)

    # 🔥 NEW: strict refusal enforcement
    if not match or score < 2:
        return REFUSAL_RESPONSE

    query_tokens = set(re.findall(r"\w+", query.lower()))
    content_tokens = set(re.findall(r"\w+", match["content"].lower()))

    # 🔥 NEW: ensure sufficient overlap
    if len(query_tokens & content_tokens) < 2:
        return REFUSAL_RESPONSE

    content = match["content"]
    sentences = re.split(r'(?<=[.!?])\s+', content)
    answer = " ".join(sentences[:2])

    return f"""{answer}

Source: {match['doc_name']}, Section {match['section']}"""


# -----------------------------
# CLI
# -----------------------------
def interactive_cli(docs):
    print("Ask My Documents (UC-X)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input(">> ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        print("\n" + answer_question(query, docs) + "\n")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    docs = load_documents()

    if args.query:
        print(answer_question(args.query, docs))
    else:
        interactive_cli(docs)


if __name__ == "__main__":
    main()