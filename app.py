import os
import re

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .source-box {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366f1;
        margin-bottom: 0.8rem;
    }

    .source-box p {
        margin: 0.3rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📚 PDF RAG Assistant</h1>
        <p>
            Upload a PDF, build a FAISS knowledge base,
            and ask questions using Retrieval-Augmented Generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=1000,
    chunk_overlap=200,
):

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(
            page_data["text"]
        )

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def create_vector_store(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    top_k=5,
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks)),
    )

    retrieved_chunks = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position == -1:
            continue

        retrieved_chunks.append(
            {
                "text": chunks[index_position]["text"],
                "page": chunks[index_position]["page"],
                "score": float(score),
            }
        )

    return retrieved_chunks


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
):

    client = get_groq_client()

    if client is None:

        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved PDF context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context, clearly say:
   "I could not find the answer in the uploaded document."
3. Give a clear and concise answer.
4. When possible, mention the relevant page number.
5. Do not use outside knowledge to fill missing information.
"""

    user_prompt = f"""
Retrieved PDF Context:

{context}

User Question:

{question}

Answer the question based only on the retrieved context.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0.2,
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
    )

    st.divider()

    st.subheader("🔑 API Status")

    if os.getenv("GROQ_API_KEY"):

        st.success(
            "Groq API key detected."
        )

    else:

        st.warning(
            "GROQ_API_KEY not detected."
        )

    st.divider()

    st.caption(
        "Embedding model: all-MiniLM-L6-v2"
    )

    st.caption(
        "Vector database: FAISS"
    )

    st.caption(
        "LLM: openai/gpt-oss-120b"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.pdf_name
        != uploaded_file.name
    ):

        st.session_state.vector_store = None
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.pdf_name = (
            uploaded_file.name
        )

    st.info(
        f"Selected document: {uploaded_file.name}"
    )

    if st.session_state.vector_store is None:

        if st.button(
            "🚀 Process PDF",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.status(
                    "Building RAG knowledge base...",
                    expanded=True,
                ):

                    st.write(
                        "📖 Extracting PDF text..."
                    )

                    pages = extract_pdf_text(
                        uploaded_file
                    )

                    if not pages:

                        st.error(
                            "No readable text was found "
                            "in this PDF."
                        )

                        st.stop()

                    st.write(
                        f"✅ Extracted text from "
                        f"{len(pages)} page(s)."
                    )

                    st.write(
                        "✂️ Creating text chunks..."
                    )

                    chunks = create_chunks(
                        pages,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    if not chunks:

                        st.error(
                            "No text chunks were created."
                        )

                        st.stop()

                    st.write(
                        f"✅ Created {len(chunks)} chunks."
                    )

                    st.write(
                        "🧠 Creating embeddings..."
                    )

                    index = create_vector_store(
                        chunks
                    )

                    st.write(
                        "✅ Embeddings stored in FAISS."
                    )

                    st.session_state.chunks = chunks

                    st.session_state.vector_store = (
                        index
                    )

                    st.success(
                        "🎉 PDF processed successfully!"
                    )

            except Exception as e:

                st.error(
                    f"Processing failed: {str(e)}"
                )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("📊 Knowledge Base")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Text Chunks",
            value=len(
                st.session_state.chunks
            ),
        )

    with col2:

        pages_count = len(
            set(
                chunk["page"]
                for chunk in
                st.session_state.chunks
            )
        )

        st.metric(
            label="Pages",
            value=pages_count,
        )

    with col3:

        st.metric(
            label="Vector Database",
            value="FAISS",
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("💬 Ask Questions")

    st.caption(
        f"Searching inside: "
        f"{st.session_state.pdf_name}"
    )

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"]
                == "assistant"
                and "sources" in message
            ):

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for source in message[
                        "sources"
                    ]:

                        st.markdown(
                            f"**Page {source['page']}**"
                        )

                        st.caption(
                            f"Similarity: "
                            f"{source['score']:.3f}"
                        )

                        st.write(
                            source["text"]
                        )

                        st.divider()

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message(
            "assistant"
        ):

            try:

                with st.spinner(
                    "🔎 Searching the document..."
                ):

                    retrieved_chunks = (
                        retrieve_chunks(
                            question,
                            st.session_state.vector_store,
                            st.session_state.chunks,
                            top_k=top_k,
                        )
                    )

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks,
                    }
                )

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for source in retrieved_chunks:

                        st.markdown(
                            f"**Page {source['page']}**"
                        )

                        st.caption(
                            f"Similarity: "
                            f"{source['score']:.3f}"
                        )

                        st.write(
                            source["text"]
                        )

                        st.divider()

            except Exception as e:

                error_message = (
                    f"Error: {str(e)}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# ============================================================
# HOW IT WORKS
# ============================================================

if st.session_state.vector_store is None:

    st.divider()

    st.subheader("🔄 How It Works")

    step1, step2 = st.columns(2)

    with step1:

        st.markdown(
            """
            **1. 📤 Upload**

            Upload your PDF document.

            **2. 📖 Extract**

            Extract readable text from the PDF.

            **3. ✂️ Chunk**

            Split the extracted text into smaller pieces.

            **4. 🧠 Embed**

            Convert each chunk into a vector embedding.
            """
        )

    with step2:

        st.markdown(
            """
            **5. 🗂️ Store**

            Store embeddings inside the FAISS vector database.

            **6. 🔎 Retrieve**

            Find the most relevant chunks for your question.

            **7. 🤖 Generate**

            Send the retrieved context to the Groq-hosted LLM
            and generate the final answer.
            """
        )

    st.info(
        "💡 Upload a PDF above to start building your knowledge base."
    )

