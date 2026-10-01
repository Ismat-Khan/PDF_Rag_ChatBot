# 📚 PDF RAG Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions about their content.

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDFs
- ✂️ Split text into overlapping chunks
- 🧠 Generate embeddings using an open-source embedding model
- 🗂️ Store embeddings using FAISS
- 🔎 Retrieve relevant document chunks
- 🤖 Generate answers using Groq's `openai/gpt-oss-120b` model
- 📚 Display retrieved source pages
- 💬 Interactive Streamlit chat interface

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — Frontend
- **PyPDF** — PDF text extraction
- **Sentence Transformers** — Text embeddings
- **FAISS** — Vector database
- **Groq API** — LLM inference
- **openai/gpt-oss-120b** — Language model

## 🔄 RAG Pipeline

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Store
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
Groq LLM
    ↓
Answer + Sources
```

## 📁 Project Structure

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/pdf-rag-chatbot.git
```

Move into the project directory:

```bash
cd pdf-rag-chatbot
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🔑 API Key

This project uses the Groq API.

Set your API key as an environment variable:

```bash
GROQ_API_KEY=your_api_key
```

For Streamlit Cloud, add the key through:

**App Settings → Secrets**

```toml
GROQ_API_KEY = "your_api_key"
```

Do not upload your API key directly to GitHub.

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📖 How It Works

1. Upload a PDF document.
2. The application extracts text from the PDF.
3. The extracted text is divided into smaller overlapping chunks.
4. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
5. The embeddings are stored in a FAISS vector index.
6. When the user asks a question, the question is also converted into an embedding.
7. FAISS searches for the most relevant chunks.
8. The retrieved context is sent to the Groq-hosted LLM.
9. The model generates an answer based on the retrieved PDF content.
10. Relevant source pages are displayed with the answer.

## ☁️ Deployment

This application can be deployed on **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Deploy the application.
6. Add your `GROQ_API_KEY` through Streamlit Secrets.

## ⚠️ Current Limitations

- The application currently works best with text-based PDFs.
- Scanned/image-only PDFs may require OCR.
- The FAISS index is stored in application memory and is rebuilt when the application restarts.
- Large documents may require additional optimization.

## 🎯 Future Improvements

- OCR support for scanned PDFs
- Persistent vector database
- Multiple PDF support
- Document history
- Better conversational memory
- Streaming responses
- Improved chunking strategies
- PDF page previews
- Authentication

## 👩‍💻 Author

**Ismat Khalil**

Built as a practical project to understand Retrieval-Augmented Generation (RAG), vector databases, embeddings, and LLM-powered applications.
