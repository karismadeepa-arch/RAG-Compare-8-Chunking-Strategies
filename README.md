# RAG: Comparison of 8 Chunking Strategies

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) ingestion and retrieval pipeline to compare eight different text chunking strategies.

Chunking is the process of dividing large documents into smaller pieces before indexing and retrieval. Choosing an appropriate chunking strategy can improve retrieval relevance, preserve context, and make information easier to find.

The project provides a Streamlit web interface where users can inspect chunks, search documents, and compare retrieval performance across different strategies.

## 2. Objectives

- Implement eight different chunking strategies.
- Build a document ingestion pipeline.
- Preserve document metadata such as title, department, and document type.
- Retrieve relevant chunks using TF-IDF and cosine similarity.
- Compare retrieval results using sample evaluation questions.
- Visualize chunk counts, search results, and evaluation metrics in Streamlit.

## 3. Technologies Used

- **Python** — Core programming language
- **Streamlit** — Interactive web interface
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **LangChain Text Splitters** — Recursive text splitting
- **Pandas** — Evaluation result tables
- **OpenAI API (optional)** — LLM-based chunking

## 4. Chunking Strategies

### 1. Fixed-Size Chunking
Divides text into chunks of a predefined character length with overlapping content to preserve context between adjacent chunks.

### 2. Recursive Chunking
Splits text using a hierarchy of separators, such as paragraphs, lines, sentences, and spaces, while attempting to preserve meaningful text boundaries.

### 3. Document-Based Chunking
Treats each complete document or section as a single chunk.

### 4. Semantic Chunking
Groups adjacent sentences according to their textual similarity using TF-IDF vectors and cosine similarity.

### 5. Query-Aware Chunking
Generates chunks dynamically based on a user's question by selecting relevant sentences and their neighboring context.

### 6. Metadata-Aware Chunking
Creates chunks while preserving metadata such as document title, department, document type, and chunk number.

### 7. LLM-Based Chunking
Uses a language model to divide documents into coherent, contextually meaningful chunks when an OpenAI API key is configured. A local sentence-grouping fallback is provided when the API is unavailable.

### 8. Agentic Chunking
Uses a rule-based routing approach to select document-based, recursive, or semantic chunking according to document length and structure.

**Note:** The current agentic implementation uses predefined routing rules rather than an autonomous AI agent.

## 5. Project Structure

```text
RAG 8 techniques/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit frontend and interactive dashboard.
- `main.py` — Document ingestion, chunking strategies, retrieval, and evaluation.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Visual Studio Code
- Internet connection for installing dependencies

### Step 1: Open the project

Open the project folder in Visual Studio Code and open a new terminal.

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

If pip is not available, run:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

Then install the requirements again.

### Step 3: Start the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser to access the application.

## 7. Application Features

### Overview
Displays sample documents, document metadata, and the number of chunks created by each strategy.

### Search & Retrieval
Allows users to enter a question, choose a chunking strategy, filter by department, and view the most relevant results with similarity scores.

### Ingested Chunks
Displays individual chunks, section names, chunk IDs, text content, department information, and available metadata.

### Evaluation
Runs predefined questions against all eight strategies and displays retrieval results, top-2 accuracy, and a strategy comparison chart.

## 8. Retrieval Method

The project uses Term Frequency–Inverse Document Frequency (TF-IDF) to represent text numerically.

Cosine similarity measures the similarity between the query vector and the chunk vectors. Chunks with positive similarity scores are ranked, and the highest-ranked results are returned.

The evaluation uses **Top-2 Hit Accuracy**: a question is counted as a hit when the expected document section appears among the first two retrieved results.

## 9. Sample Use Case

**Question:** Who approves employee leave requests?

**Expected answer source:** Leave Policy

The system searches the selected strategy's chunks and ranks relevant text. The result should identify the Leave Policy section containing the statement that managers approve leave requests.

The project includes sample HR, attendance, IT security, and expense reimbursement documents.

## 10. Configuration for LLM-Based Chunking

The project can use OpenAI for LLM-based chunking if an API key is available.

In Windows Command Prompt, set the key for the current terminal session:

```cmd
set OPENAI_API_KEY=your_api_key_here
```

Then start the application:

```cmd
python -m streamlit run app.py
```

Without an API key, the LLM-based strategy uses the included local fallback. The API key should be kept private and must not be committed to GitHub.

## 11. Limitations

- The project uses sample documents rather than a large real-world knowledge base.
- TF-IDF is a lexical retrieval method and does not understand semantic meaning as deeply as embedding models.
- Semantic chunking uses TF-IDF similarity as a lightweight approximation.
- LLM-based chunking depends on API availability when enabled.
- Agentic chunking currently uses predefined rules.
- Evaluation results depend on the sample documents and predefined questions.
- The project does not yet include a persistent vector database such as ChromaDB.

## 12. Future Enhancements

- Add document upload support for PDF, TXT, and DOCX files.
- Integrate sentence-transformer embeddings.
- Store chunks in ChromaDB or another vector database.
- Add a larger evaluation dataset and metrics such as Recall@K and Mean Reciprocal Rank.
- Implement a more advanced agentic chunking workflow.
- Add answer generation using a language model and retrieved context.
- Support configurable chunk sizes and overlap values.

## 13. Conclusion

This project provides a practical demonstration of eight chunking approaches and their effects on text retrieval. The Streamlit interface makes it possible to inspect generated chunks, search documents, and compare retrieval performance.

It serves as a foundation for further development into a more complete RAG application using embedding models, vector databases, and LLM-generated answers.

## Author

Developed as a learning project to explore document processing, chunking strategies, information retrieval, and RAG application development.
