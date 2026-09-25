import os
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple

from pypdf import PdfReader
import chromadb
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_TITLE = "PDF RAG Chatbot"
GROQ_MODEL = "openai/gpt-oss-120b"

# Multilingual embedding model: English + Urdu and many other languages.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:
    """
    Reads the Groq API key.

    Priority:
    1. Streamlit secrets: Pdf_API_key
    2. Environment variable: Pdf_API_key
    3. Environment variable: GROQ_API_KEY
    """

    try:
        if "Pdf_API_key" in st.secrets:
            return st.secrets["Pdf_API_key"]
    except Exception:
        pass

    return (
        os.getenv("Pdf_API_key")
        or os.getenv("GROQ_API_KEY")
        or ""
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text page-by-page from the uploaded PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pages = []

    reader = PdfReader(BytesIO(pdf_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", " ").strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Split each PDF page into smaller overlapping chunks.

    Page number is preserved in metadata so the chatbot can
    show the source page with every answer.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "۔ ",
            ". ",
            "؟ ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = splitter.split_text(page_text)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "id": f"page_{page_number}_chunk_{chunk_number}",
                    "text": chunk_text,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

    return chunks


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_vector_store(chunks: List[Dict]):
    """
    Create an in-memory ChromaDB collection for the current PDF.
    """

    embedding_model = load_embedding_model()

    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    client = chromadb.Client()

    # Unique collection name for this Streamlit session/PDF.
    collection_name = "pdf_rag_collection"

    # If an old collection exists in this process, remove it.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[item["id"] for item in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "page": item["page"],
                "chunk": item["chunk"],
            }
            for item in chunks
        ],
    )

    return client, collection


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    collection,
    question: str,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a question.
    """

    embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "page": metadata.get("page", "?"),
                "chunk": metadata.get("chunk", "?"),
                "distance": distance,
            }
        )

    return retrieved


# ============================================================
# CONTEXT CREATION
# ============================================================

def build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Convert retrieved chunks into a context block for the LLM.
    """

    context_parts = []

    for index, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {index} | Page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client() -> Groq:
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "Groq API key not found. Set Pdf_API_key in "
            "Streamlit secrets or as an environment variable."
        )

    return Groq(api_key=api_key)


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Answer the user's question using only retrieved PDF context.
    """

    context = build_context(retrieved_chunks)
    client = get_groq_client()

    system_prompt = """
You are a document question-answering assistant.

Your task is to answer questions using ONLY the information
contained in the provided document context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge to fill missing information.
3. If the answer is not supported by the context, clearly say:
   "The answer was not found in the uploaded document."
4. Answer in the same language as the user's question.
5. If the user asks in Urdu, answer in clear Urdu.
6. If the user asks in English, answer in English.
7. Give a concise but sufficiently detailed answer.
8. When possible, mention the relevant PDF page number.
9. If the question asks for a list, preserve the document's terminology.
10. Distinguish between what the document explicitly states and
    what cannot be determined from the provided context.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

Answer the question from the document context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
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
        temperature=0.1,
        max_completion_tokens=2000,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

def summarize_batch(
    chunks: List[Dict],
    language_instruction: str,
) -> str:
    """
    Summarize one batch of document chunks.
    """

    client = get_groq_client()

    batch_text = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in chunks
    )

    prompt = f"""
You are summarizing part of an uploaded PDF.

{language_instruction}

Summarize ONLY the material provided below.

Requirements:
- Preserve important facts, rules, names, dates, sections,
  conditions, exceptions, procedures, and terminology.
- Do not add information from outside the document.
- Avoid repeating the same point.
- Keep page references where useful.
- Produce a structured summary.

DOCUMENT EXCERPT:
-----------------
{batch_text}
-----------------
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document summarization assistant. "
                    "Use only the supplied document text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_completion_tokens=2500,
    )

    return response.choices[0].message.content.strip()


def summarize_summaries(
    partial_summaries: List[str],
    language_instruction: str,
) -> str:
    """
    Combine intermediate summaries into one final document summary.
    """

    client = get_groq_client()

    combined = "\n\n".join(
        f"PARTIAL SUMMARY {i + 1}:\n{summary}"
        for i, summary in enumerate(partial_summaries)
    )

    prompt = f"""
Create a final structured summary of the uploaded PDF from the
partial summaries below.

{language_instruction}

Requirements:
- Use ONLY the supplied partial summaries.
- Preserve important rules, sections, dates, procedures,
  conditions, exceptions, and terminology.
- Do not introduce outside information.
- Organize the result with clear headings and bullet points.
- Avoid unnecessary repetition.
- Make the summary useful for someone who wants to understand
  the document without reading every page.
- If page numbers are present in the partial summaries, preserve
  them where useful.

PARTIAL SUMMARIES:
------------------
{combined}
------------------
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document synthesis assistant. "
                    "Do not add information not present in the supplied summaries."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_completion_tokens=5000,
    )

    return response.choices[0].message.content.strip()


def summarize_document(
    chunks: List[Dict],
    language: str = "English",
) -> str:
    """
    Hierarchical summarization:
        PDF chunks -> partial summaries -> final summary

    This avoids putting the entire PDF into one LLM request.
    """

    if language == "Urdu":
        language_instruction = (
            "Write the summary in clear Urdu. Keep important official "
            "English terms in English in parentheses when appropriate."
        )
    else:
        language_instruction = (
            "Write the summary in clear professional English."
        )

    # Larger batches reduce the number of Groq requests.
    batch_size = 30

    partial_summaries = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    progress = st.progress(0)
    status = st.empty()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        current_batch = (start // batch_size) + 1
        status.info(
            f"Creating summary: batch {current_batch} of {total_batches}..."
        )

        summary = summarize_batch(
            batch,
            language_instruction,
        )

        partial_summaries.append(summary)

        progress.progress(
            min(current_batch / total_batches, 1.0)
        )

    status.info("Combining partial summaries...")

    # If there is only one batch, no second LLM call is necessary.
    if len(partial_summaries) == 1:
        final_summary = partial_summaries[0]
    else:
        # If there are many summaries, combine them recursively.
        current_summaries = partial_summaries

        while len(current_summaries) > 8:
            grouped = []

            for start in range(0, len(current_summaries), 8):
                group = current_summaries[start:start + 8]

                group_text = "\n\n".join(
                    f"SUMMARY {i + 1}:\n{text}"
                    for i, text in enumerate(group)
                )

                client = get_groq_client()

                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Combine the supplied document summaries "
                                "without adding outside information."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"""
{language_instruction}

Combine the following summaries into one coherent summary.
Preserve important facts, rules, dates, procedures, conditions,
exceptions, and terminology.

{group_text}
""",
                        },
                    ],
                    temperature=0.1,
                    max_completion_tokens=3500,
                            )

                grouped.append(
                    response.choices[0].message.content.strip()
                )

            current_summaries = grouped

        final_summary = summarize_summaries(
            current_summaries,
            language_instruction,
        )

    progress.empty()
    status.empty()

    return final_summary


# ============================================================
# STREAMLIT SESSION STATE
# ============================================================

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None

    if "pages" not in st.session_state:
        st.session_state.pages = []

    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    if "collection" not in st.session_state:
        st.session_state.collection = None

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""

    if "summary" not in st.session_state:
        st.session_state.summary = None


initialize_session_state()


# ============================================================
# UI HEADER
# ============================================================

st.title("📚 PDF RAG Chatbot")

st.markdown(
    """
Upload a PDF and ask questions about its contents.

**Features**
- 📄 PDF upload
- 🔎 RAG-based question answering
- 🌐 English + Urdu support
- 🧠 Multilingual embeddings
- 📑 Page/source references
- 📝 Whole-document summary
- ⚡ Groq `openai/gpt-oss-120b`
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a text-based PDF for question answering and summarization.",
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=3,
        max_value=10,
        value=TOP_K,
        step=1,
    )

    st.divider()

    st.info(
        "For local development, set Pdf_API_key in a .env file. "
        "For Streamlit Cloud, add Pdf_API_key under App Settings → Secrets."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = None
        st.rerun()


# ============================================================
# NO PDF
# ============================================================

if uploaded_file is None:
    st.info("👈 Upload a PDF from the sidebar to begin.")

    st.markdown(
        """
### Example questions

- What is this document about?
- What are the main rules?
- What are the conditions for promotion?
- Which authority is responsible for this procedure?
- Summarize this document.
- Explain this section in simple Urdu.
"""
    )

    st.stop()


# ============================================================
# PROCESS NEW PDF
# ============================================================

pdf_bytes = uploaded_file.getvalue()
current_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.pdf_hash != current_hash:

    st.session_state.pdf_hash = current_hash
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.messages = []
    st.session_state.summary = None
    st.session_state.collection = None
    st.session_state.pages = []
    st.session_state.chunks = []

    with st.spinner("Extracting text from PDF..."):
        pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        st.error(
            "No selectable text was found in this PDF. "
            "This version of the app does not include OCR yet."
        )
        st.stop()

    with st.spinner("Creating document chunks..."):
        chunks = create_chunks(pages)

    if not chunks:
        st.error("No usable text chunks were created from the PDF.")
        st.stop()

    with st.spinner("Creating multilingual vector database..."):
        _, collection = create_vector_store(chunks)

    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.collection = collection


# ============================================================
# PDF INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "PDF Pages",
        len(st.session_state.pages),
    )

with col2:
    st.metric(
        "Text Chunks",
        len(st.session_state.chunks),
    )

with col3:
    st.metric(
        "Model",
        "GPT-OSS 120B",
    )

st.caption(f"Current document: **{st.session_state.pdf_name}**")


# ============================================================
# TABS
# ============================================================

tab_chat, tab_summary = st.tabs(
    ["💬 Ask Questions", "📝 Summarize PDF"]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab_chat:

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📑 Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **Page {source['page']}**, "
                            f"chunk {source['chunk']}"
                        )

    question = st.chat_input(
        "Ask a question about the uploaded PDF..."
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

            with st.spinner("Searching the document..."):

                retrieved = retrieve_chunks(
                    st.session_state.collection,
                    question,
                    top_k=top_k,
                )

            if not retrieved:
                answer = (
                    "I could not find relevant information in the "
                    "uploaded document."
                )
                sources = []

            else:
                with st.spinner("Generating answer..."):
                    try:
                        answer = answer_question(
                            question,
                            retrieved,
                        )
                    except Exception as error:
                        answer = (
                            "An error occurred while generating the answer:\n\n"
                            f"`{error}`"
                        )

                sources = [
                    {
                        "page": item["page"],
                        "chunk": item["chunk"],
                    }
                    for item in retrieved
                ]

            st.markdown(answer)

            if sources:
                with st.expander("📑 Sources used"):
                    unique_sources = []
                    seen = set()

                    for source in sources:
                        key = (
                            source["page"],
                            source["chunk"],
                        )

                        if key not in seen:
                            seen.add(key)
                            unique_sources.append(source)

                    for source in unique_sources:
                        st.markdown(
                            f"- Page **{source['page']}**, "
                            f"chunk **{source['chunk']}**"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


# ============================================================
# SUMMARY TAB
# ============================================================

with tab_summary:

    st.subheader("Document Summary")

    summary_language = st.radio(
        "Summary language",
        ["English", "Urdu"],
        horizontal=True,
    )

    if st.button(
        "📝 Generate Full PDF Summary",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner(
                "Summarizing the document. Large PDFs may take some time..."
            ):
                st.session_state.summary = summarize_document(
                    st.session_state.chunks,
                    language=summary_language,
                )

        except Exception as error:
            st.error(
                f"Could not generate the summary: {error}"
            )

    if st.session_state.summary:
        st.markdown(st.session_state.summary)

        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="pdf_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RAG pipeline: PDF → text extraction → chunking → multilingual "
    "embeddings → ChromaDB → retrieval → Groq GPT-OSS 120B"
)
