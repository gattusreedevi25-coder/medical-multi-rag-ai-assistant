
import os
import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

st.set_page_config(
    page_title="Medical Multi-RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Multi-RAG AI Assistant")
st.write(
    "Upload medical PDF documents and ask questions "
    "about their contents."
)

st.warning(
    "Educational use only. This application does not diagnose "
    "medical conditions or replace professional medical advice."
)

# Load the embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Split document text into smaller chunks
def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# Read multiple PDF files
def process_pdfs(files):
    documents = []

    for file in files:
        try:
            reader = PdfReader(file)

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                for chunk in split_text(text):
                    documents.append({
                        "source": file.name,
                        "page": page_number,
                        "text": chunk
                    })

        except Exception as error:
            st.warning(f"Could not read {file.name}: {error}")

    return documents

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Number of passages to retrieve",
        min_value=1,
        max_value=5,
        value=3
    )

    st.caption("Upload multiple PDFs to search across documents.")

# Upload documents
uploaded_files = st.file_uploader(
    "Upload medical PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📚 Process Documents", type="primary"):
        with st.spinner("Reading PDFs and creating embeddings..."):
            documents = process_pdfs(uploaded_files)

            if not documents:
                st.error(
                    "No readable text was found. "
                    "Scanned PDFs may require OCR."
                )
            else:
                model = load_embedding_model()
                texts = [doc["text"] for doc in documents]
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                st.session_state["documents"] = documents
                st.session_state["embeddings"] = embeddings

                st.success(
                    f"Processed {len(uploaded_files)} PDF file(s) "
                    f"and created {len(documents)} text passages."
                )

# Ask questions after processing
if "documents" in st.session_state:
    st.divider()
    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What does the document say about diabetes?"
    )

    if st.button("🔎 Search and Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error(
                "GROQ_API_KEY is missing. Add it to your environment "
                "variables in Render."
            )
        else:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    model = load_embedding_model()
                    question_embedding = model.encode(
                        [question],
                        normalize_embeddings=True
                    )

                    similarities = cosine_similarity(
                        question_embedding,
                        st.session_state["embeddings"]
                    )[0]

                    best_indices = np.argsort(similarities)[::-1][:top_k]

                    context_parts = []
                    sources = []

                    for index in best_indices:
                        doc = st.session_state["documents"][index]

                        context_parts.append(
                            f"Source: {doc['source']}, "
                            f"Page: {doc['page']}\n{doc['text']}"
                        )

                        source_info = (
                            doc["source"],
                            doc["page"]
                        )

                        if source_info not in sources:
                            sources.append(source_info)

                    context = "\n\n".join(context_parts)

                    client = Groq(
                        api_key=os.environ["GROQ_API_KEY"]
                    )

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a medical document assistant. "
                                    "Answer using only the provided context. "
                                    "If the context does not contain the answer, "
                                    "say that the uploaded documents do not "
                                    "provide enough information. Do not diagnose "
                                    "or prescribe treatment. Mention that your "
                                    "answer is educational information only."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {question}\n\n"
                                    f"Retrieved document context:\n{context}"
                                )
                            }
                        ],
                        temperature=0.2,
                        max_tokens=800
                    )

                    st.subheader("🤖 AI Answer")
                    st.write(response.choices[0].message.content)

                    st.subheader("📑 Retrieved Sources")

                    for source, page in sources:
                        st.write(f"- **{source}** — Page {page}")

                    with st.expander("View retrieved passages"):
                        st.text(context)

                except Exception as error:
                    st.error(
                        "Unable to generate an answer. Check your API key, "
                        "model access, and connection."
                    )
                    st.caption(str(error))

st.divider()
st.caption(
    "Multi-document retrieval demo | Python | Streamlit | "
    "Sentence Transformers | Groq"
)


