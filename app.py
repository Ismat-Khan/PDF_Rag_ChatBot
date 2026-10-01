import os
import re
import tempfile

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


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
        .main {
            background-color: #f8fafc;
        }

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
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
        }

        .info-card {
            padding: 1.2rem;
            border-radius: 16px;
            background-color: white;
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }

        .source-card {
            padding: 1rem;
            border-radius: 12px;
            background-color: #f1f5f9;
            border-left: 4px solid #6366f1;
            margin-bottom: 0.8rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background-color: white;
            border: 1px solid #e2e8f0;
            text-align: center;
        }

        .small-text {
            color: #64748b;
            font-size: 0.9rem;
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
            Upload a PDF, build a FAISS knowledge base, and ask questions
            using retrieval-augmented generation.
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
    """
    Loads an open-source embedding model.

    The model runs locally on the Streamlit server.
    No embedding API key is required.
    """
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
    """
    Extracts text from every page of the uploaded PDF.
    """

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
    """
    Cleans unnecessary whitespace from extracted PDF text.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(pages, chunk_size=1000, chunk_overlap=200):
    """
    Creates overlapping text chunks.

    Each chunk keeps its original PDF page number.
    """

    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = clean_text(page_data["text"])

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
# CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vector_store(chunks):
    """
    Converts chunks into embeddings and stores them in FAISS.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(question, index, chunks, top_k=5):
    """
    Converts the question into an embedding and retrieves
    the most relevant chunks from FAISS.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks)),
    )

    retrieved_chunks = []

    for score, index_position in zip(scores[0], indices[0]):

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

def generate_answer(question, retrieved_chunks):
    """
    Sends the retrieved context to the Groq model.
    """

    client = get_groq_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is not configured."
        )

    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

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
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks provide more context but may increase the prompt size.",
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

    st.subheader("🔑 API Key")

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key detected.")
    else:
        st.warning(
            "Groq API key not detected. Add GROQ_API_KEY to Streamlit Secrets."
        )

    st.divider()

    st.caption(
        "Embeddings: all-MiniLM-L6-v2\n\n"
        "Vector DB: FAISS\n\n"
        "LLM: openai/gpt-oss-120b via Groq"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if st.session_state.pdf_name != uploaded_file.name:

        st.session_state.vector_store = None
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.pdf_name = uploaded_file.name

    if st.session_state.vector_store is None:

        st.info(
            "Your PDF is ready. Click the button below to build the knowledge base."
        )

        if st.button(
            "🚀 Process PDF",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.status(
                    "Building your RAG knowledge base...",
                    expanded=True,
                ):

                    st.write("📖 Extracting PDF text...")

                    pages = extract_pdf_text(uploaded_file)

                    if not pages:
                        st.error(
                            "No readable text was found in this PDF. "
                            "If it is a scanned PDF, OCR may be required."
                        )
                        st.stop()

                    st.write(
                        f"✅ Extracted text from {len(pages)} page(s)."
                    )

                    st.write("✂️ Creating text chunks...")

                    chunks = create_chunks(
                        pages,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    if not chunks:
                        st.error(
                            "No text chunks could be created."
                        )
                        st.stop()

                    st.write(
                        f"✅ Created {len(chunks)} chunks."
                    )

                    st.write(
                        "🧠 Creating embeddings..."
                    )

                    index = create_vector_store(chunks)

                    st.write(
                        "✅ Embeddings created and stored in FAISS."
                    )

                    st.session_state.chunks = chunks
                    st.session_state.vector_store = index

                    st.success(
                        "🎉 PDF processed successfully!"
                    )

            except Exception as e:

                st.error(
                    f"Processing failed: {str(e)}"
                )


# ============================================================
# KNOWLEDGE BASE INFORMATION
# ============================================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("📊 Knowledge Base")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{len(st.session_state.chunks)}</h3>
                <div class="small-text">Text Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        pages_count = len(
            set(
                chunk["page"]
                for chunk in st.session_state.chunks
            )
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{pages_count}</h3>
                <div class="small-text">Pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>FAISS</h3>
                <div class="small-text">Vector Database</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT SECTION
# ============================================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("💬 2. Ask Questions")

    st.caption(
        f"Currently searching: {st.session_state.pdf_name}"
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("📚 View retrieved sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>Page {source["page"]}</b>
                                <br>
                                <span class="small-text">
                                    Similarity: {source["score"]:.3f}
                                </span>
                                <br><br>
                                {source["text"]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "🔎 Searching the document..."
                ):

                    retrieved_chunks = retrieve_chunks(
                        question,
                        st.session_state.vector_store,
                        st.session_state.chunks,
                        top_k=top_k,
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
                    "📚 View retrieved sources"
                ):

                    for source in retrieved_chunks:

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>Page {source["page"]}</b>
                                <br>
                                <span class="small-text">
                                    Similarity: {source["score"]:.3f}
                                </span>
                                <br><br>
                                {source["text"]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as e:

                error_message = f"Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
        <div class="info-card">
            <h3>🚀 How it works</h3>

            <p>
                <b>1. Upload</b> your PDF document.
            </p>

            <p>
                <b>2. Extract</b> the document text.
            </p>

            <p>
                <b>3. Chunk</b> the text into smaller pieces.
            </p>

            <p>
                <b>4. Embed</b> each chunk using an open-source embedding model.
            </p>

            <p>
                <b>5. Store</b> the embeddings inside FAISS.
            </p>

            <p>
                <b>6. Retrieve</b> the most relevant chunks for your question.
            </p>

            <p>
                <b>7. Generate</b> an answer using the Groq-hosted LLM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
