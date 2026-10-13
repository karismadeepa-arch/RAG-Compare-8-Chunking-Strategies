import os
import re
from collections import defaultdict
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Sample documents and questions
# -----------------------------
DOCUMENTS = [
    {
        "title": "Leave Policy",
        "department": "HR",
        "type": "policy",
        "text": """
Employees can request leave through the HRMS application.
The manager must approve leave requests.
Planned leave should be requested in advance.
Sick leave must follow the company leave procedure.
"""
    },
    {
        "title": "Attendance Policy",
        "department": "HR",
        "type": "policy",
        "text": """
Employees must record their attendance using the HRMS application.
Employees should check in when they start work and check out when they finish.
Attendance corrections must be submitted to the HR department.
Managers review attendance records regularly.
"""
    },
    {
        "title": "IT Security Policy",
        "department": "IT",
        "type": "policy",
        "text": """
Employees must keep their passwords confidential.
Multi-factor authentication should be enabled where available.
Suspicious emails must be reported to the IT department.
Company data must not be shared with unauthorized people.
"""
    },
    {
        "title": "Expense Reimbursement",
        "department": "Finance",
        "type": "procedure",
        "text": """
Employees can submit eligible business expenses for reimbursement.
Receipts must be attached to each expense claim.
The manager reviews and approves the claim before Finance processes payment.
Claims should be submitted within thirty days of the expense.
"""
    },
]

QUESTIONS = [
    {"query": "Who approves employee leave requests?", "expected_section": "Leave Policy"},
    {"query": "How do employees record attendance?", "expected_section": "Attendance Policy"},
    {"query": "What should employees do with suspicious emails?", "expected_section": "IT Security Policy"},
    {"query": "Who approves expense claims?", "expected_section": "Expense Reimbursement"},
]


# -----------------------------
# Helpers
# -----------------------------
def make_chunk(document, text, method, index):
    return {
        "id": f"{method.lower().replace(' ', '-')}-{index}",
        "section": document["title"],
        "department": document["department"],
        "document_type": document["type"],
        "method": method,
        "text": text.strip(),
    }


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def fixed_chunks(text, size=180, overlap=30):
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def recursive_split(text, size=220, overlap=35):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text.strip())


def group_sentences(sentences, max_chars=220):
    groups, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


# -----------------------------
# Strategy 1: Fixed-size
# -----------------------------
def fixed_strategy(documents):
    output = []
    for doc in documents:
        for i, piece in enumerate(fixed_chunks(doc["text"]), 1):
            output.append(make_chunk(doc, piece, "Fixed-size", i))
    return output


# -----------------------------
# Strategy 2: Recursive
# -----------------------------
def recursive_strategy(documents):
    output = []
    for doc in documents:
        for i, piece in enumerate(recursive_split(doc["text"]), 1):
            output.append(make_chunk(doc, piece, "Recursive", i))
    return output


# -----------------------------
# Strategy 3: Document-based
# -----------------------------
def document_strategy(documents):
    return [
        make_chunk(doc, doc["text"], "Document-based", 1)
        for doc in documents
    ]


# -----------------------------
# Strategy 4: Semantic
# -----------------------------
def semantic_strategy(documents, threshold=0.04):
    output = []
    for doc in documents:
        sentences = split_sentences(doc["text"])
        if len(sentences) <= 1:
            groups = sentences
        else:
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                vectors = vectorizer.fit_transform(sentences)
                groups, current = [], [sentences[0]]
                for i in range(1, len(sentences)):
                    similarity = cosine_similarity(vectors[i - 1], vectors[i])[0][0]
                    if similarity >= threshold:
                        current.append(sentences[i])
                    else:
                        groups.append(" ".join(current))
                        current = [sentences[i]]
                if current:
                    groups.append(" ".join(current))
            except ValueError:
                groups = group_sentences(sentences)
        for i, piece in enumerate(groups, 1):
            output.append(make_chunk(doc, piece, "Semantic", i))
    return output


# -----------------------------
# Strategy 5: Query-aware
# Generated for each search query
# -----------------------------
def query_aware_strategy(documents, query):
    output = []
    for doc in documents:
        sentences = split_sentences(doc["text"])
        if not sentences:
            continue
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(sentences + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        except ValueError:
            continue

        selected = set()
        for i, score in enumerate(scores):
            if score > 0:
                selected.add(i)
                if i > 0:
                    selected.add(i - 1)
                if i + 1 < len(sentences):
                    selected.add(i + 1)

        if not selected and len(sentences) > 0:
            best = int(scores.argmax())
            if scores[best] > 0:
                selected.add(best)

        if selected:
            relevant_text = " ".join(sentences[i] for i in sorted(selected))
            chunk = make_chunk(doc, relevant_text, "Query-aware", 1)
            chunk["metadata"] = {"query": query, "selected_sentence_indexes": sorted(selected)}
            output.append(chunk)
    return output


# -----------------------------
# Strategy 6: Metadata-aware
# -----------------------------
def metadata_strategy(documents):
    output = []
    for doc in documents:
        pieces = recursive_split(doc["text"])
        for i, piece in enumerate(pieces, 1):
            chunk = make_chunk(doc, piece, "Metadata-aware", i)
            chunk["metadata"] = {
                "title": doc["title"],
                "department": doc["department"],
                "document_type": doc["type"],
                "chunk_number": i,
            }
            output.append(chunk)
    return output


# -----------------------------
# Strategy 7: LLM-based
# Uses OpenAI if OPENAI_API_KEY exists;
# otherwise uses a local sentence-grouping fallback.
# -----------------------------
def llm_strategy(documents):
    output = []
    client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception:
            client = None

    for doc in documents:
        pieces = None
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Split the supplied document into coherent retrieval chunks. "
                                "Keep all important information. Return only a JSON array of strings."
                            ),
                        },
                        {
                            "role": "user",
                            "content": doc["text"],
                        },
                    ],
                    temperature=0,
                )
                import json
                parsed = json.loads(response.choices[0].message.content)
                if isinstance(parsed, list):
                    pieces = [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pieces = None

        if not pieces:
            # Offline fallback: group adjacent sentences into coherent chunks.
            pieces = group_sentences(split_sentences(doc["text"]), max_chars=220)

        for i, piece in enumerate(pieces, 1):
            chunk = make_chunk(doc, piece, "LLM-based", i)
            chunk["metadata"] = {
                "chunking_mode": "OpenAI" if client is not None and pieces else "local fallback"
            }
            output.append(chunk)
    return output


# -----------------------------
# Strategy 8: Agentic
# Rule-based router chooses a method per document.
# -----------------------------
def agentic_strategy(documents):
    output = []
    for doc in documents:
        text = doc["text"].strip()
        sentences = split_sentences(text)

        if len(text) <= 250 and len(sentences) <= 5:
            chosen = "document-based"
            pieces = [text]
        elif len(text) > 450:
            chosen = "recursive"
            pieces = recursive_split(text)
        else:
            chosen = "semantic"
            semantic_chunks = semantic_strategy([doc])
            pieces = [chunk["text"] for chunk in semantic_chunks]

        for i, piece in enumerate(pieces, 1):
            chunk = make_chunk(doc, piece, "Agentic", i)
            chunk["chosen_strategy"] = chosen
            output.append(chunk)
    return output


STRATEGY_FUNCTIONS = {
    "Fixed-size": fixed_strategy,
    "Recursive": recursive_strategy,
    "Document-based": document_strategy,
    "Semantic": semantic_strategy,
    "Query-aware": None,  # generated per query
    "Metadata-aware": metadata_strategy,
    "LLM-based": llm_strategy,
    "Agentic": agentic_strategy,
}


def build_indexes():
    """Build chunks for all static strategies, including query-aware placeholder."""
    indexes = {}
    chunk_counts = {}
    for name, function in STRATEGY_FUNCTIONS.items():
        chunks = [] if function is None else function(DOCUMENTS)
        indexes[name] = {
            "name": name,
            "chunks": chunks,
            "vectorizer": None,
            "matrix": None,
        }
        chunk_counts[name] = len(chunks)
    return indexes, chunk_counts


def create_index(chunks, name="default"):
    """Create an index while preserving chunk dictionaries."""
    normalized = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            item = chunk.copy()
            item["text"] = str(item.get("text", item.get("content", ""))).strip()
        else:
            item = {
                "id": f"{name}-{i}",
                "section": "Unknown Section",
                "department": "Not specified",
                "document_type": "unknown",
                "method": name,
                "text": str(chunk).strip(),
            }
        if item["text"]:
            normalized.append(item)

    texts = [item["text"] for item in normalized]
    vectorizer, matrix = None, None
    if texts:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            vectorizer, matrix = None, None

    return {
        "name": name,
        "chunks": normalized,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def search_index(index, query, top_k=3):
    if not index.get("chunks") or index.get("vectorizer") is None or index.get("matrix") is None:
        return []
    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for i in ranked:
        if scores[i] > 0:
            result = index["chunks"][i].copy()
            result["score"] = float(scores[i])
            results.append(result)
    return results


def retrieve(chunks, query, top_k=2, department=None):
    eligible = [
        chunk for chunk in chunks
        if department is None
        or str(chunk.get("department", "")).lower() == department.lower()
    ]
    if not eligible:
        return []

    searchable_text = []
    for chunk in eligible:
        metadata = chunk.get("metadata", {})
        metadata_text = " ".join(str(value) for value in metadata.values())
        searchable_text.append(f'{chunk.get("text", "")} {metadata_text}')

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(searchable_text)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, vectors).flatten()
    except ValueError:
        return []

    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        result = eligible[idx].copy()
        result["score"] = float(scores[idx])
        results.append(result)
    return results


def evaluate(indexes, questions=QUESTIONS):
    rows = []
    strategy_names = list(STRATEGY_FUNCTIONS.keys())
    for item in questions:
        query = item["query"]
        expected = item["expected_section"]
        query_chunks = query_aware_strategy(DOCUMENTS, query)
        for name in strategy_names:
            chunks = query_chunks if name == "Query-aware" else indexes[name]["chunks"]
            results = retrieve(chunks, query, top_k=2)
            found = any(result.get("section") == expected for result in results)
            rows.append({
                "question": query,
                "strategy": name,
                "expected_section": expected,
                "top_2_hit": found,
                "result_count": len(results),
                "top_result": results[0].get("section", "") if results else "",
                "top_score": results[0].get("score", 0.0) if results else 0.0,
            })
    return rows
