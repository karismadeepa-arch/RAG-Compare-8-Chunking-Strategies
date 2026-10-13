import streamlit as st
import pandas as pd
import main as rag

st.set_page_config(page_title="RAG: 8 Chunking Strategies", page_icon="🔎", layout="wide")

st.title("RAG: Compare 8 Chunking Strategies")
st.caption("Document ingestion • TF-IDF retrieval • Strategy comparison")

STRATEGIES = list(rag.STRATEGY_FUNCTIONS.keys())

@st.cache_resource
def load_indexes():
    return rag.build_indexes()

indexes, chunk_counts = load_indexes()

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Search & Retrieval", "Ingested Chunks", "Evaluation"
])

with tab1:
    st.subheader("Ingestion overview")
    cols = st.columns(4)
    for i, name in enumerate(STRATEGIES):
        count = chunk_counts.get(name, 0)
        cols[i % 4].metric(name, "Query-based" if name == "Query-aware" else count)

    st.markdown("### Documents")
    st.dataframe(
        pd.DataFrame([
            {
                "Title": doc["title"],
                "Department": doc["department"],
                "Type": doc["type"],
                "Characters": len(doc["text"].strip())
            }
            for doc in rag.DOCUMENTS
        ]),
        use_container_width=True,
        hide_index=True
    )
    st.info(
        "LLM-based chunking uses OpenAI when OPENAI_API_KEY is configured. "
        "Without a key, it uses a local sentence-grouping fallback."
    )

with tab2:
    st.subheader("Search documents")
    query = st.text_input(
        "Enter your question",
        value="Who approves employee leave requests?",
        key="search_query"
    )
    selected_strategy = st.selectbox(
        "Select chunking strategy",
        STRATEGIES,
        key="search_strategy"
    )
    department = st.selectbox(
        "Department filter",
        ["All", "HR", "IT", "Finance"],
        key="department_filter"
    )
    top_k = st.slider("Number of results", min_value=1, max_value=5, value=3)

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            if selected_strategy == "Query-aware":
                chunks = rag.query_aware_strategy(rag.DOCUMENTS, query)
            else:
                chunks = indexes[selected_strategy]["chunks"]

            results = rag.retrieve(
                chunks, query, top_k=top_k,
                department=None if department == "All" else department
            )
            if not results:
                st.warning("No matching chunks found. Try a different question.")
            else:
                for rank, result in enumerate(results, 1):
                    with st.expander(
                        f"#{rank} | {result.get('section', 'Unknown')} | "
                        f"Score: {result.get('score', 0):.3f}",
                        expanded=(rank == 1)
                    ):
                        st.write(result.get("text", ""))
                        st.caption(
                            f"ID: {result.get('id', 'N/A')} • "
                            f"Department: {result.get('department', 'N/A')} • "
                            f"Method: {result.get('method', selected_strategy)}"
                        )
                        if result.get("metadata"):
                            st.json(result["metadata"])
                        if result.get("chosen_strategy"):
                            st.write("Agent selected:", result["chosen_strategy"])

with tab3:
    st.subheader("View ingested chunks")
    chosen = st.selectbox(
        "Select strategy to inspect",
        STRATEGIES,
        key="chunk_strategy"
    )
    if chosen == "Query-aware":
        query_for_chunks = st.text_input(
            "Enter a question to generate query-aware chunks",
            value="Who approves employee leave requests?",
            key="query_aware_inspect"
        )
        chunks = rag.query_aware_strategy(rag.DOCUMENTS, query_for_chunks)
    else:
        chunks = indexes[chosen]["chunks"]

    st.metric("Total chunks", len(chunks))
    if not chunks:
        st.warning("No chunks found for this strategy.")
    for i, chunk in enumerate(chunks):
        section = chunk.get("section", "Unknown section") if isinstance(chunk, dict) else "Text chunk"
        chunk_id = chunk.get("id", i) if isinstance(chunk, dict) else i
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        with st.expander(f"{section} — {chunk_id}"):
            st.write(text)
            if isinstance(chunk, dict):
                st.caption(
                    f"Department: {chunk.get('department', 'Not specified')} • "
                    f"Type: {chunk.get('document_type', 'Not specified')}"
                )
                if chunk.get("metadata"):
                    st.json(chunk["metadata"])
                if chunk.get("chosen_strategy"):
                    st.write("Agent selected:", chunk["chosen_strategy"])

with tab4:
    st.subheader("Compare retrieval results")
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating all strategies..."):
            rows = rag.evaluate(indexes)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        summary = (
            df.groupby("strategy", as_index=False)
            .agg(
                Questions=("question", "count"),
                Top_2_Hits=("top_2_hit", "sum"),
                Average_Top_Score=("top_score", "mean"),
            )
        )
        summary["Top_2_Accuracy_%"] = (
            summary["Top_2_Hits"] / summary["Questions"] * 100
        ).round(1)
        st.markdown("### Strategy summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("strategy")["Top_2_Accuracy_%"])
