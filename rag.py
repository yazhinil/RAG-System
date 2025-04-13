import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from IPython.display import Markdown, display
import ipywidgets as widgets
import shutil
from langchain_chroma import Chroma
import numpy
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
CHROMA_PATH = "/content/chroma"
DATA_PATH = "/content/rag"  # Path to your markdown files
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.7
TOP_K_RESULTS = 3

PROMPT_TEMPLATE = """
Answer the question based only on the following medical oncology context:

{context}

---

Question: {question}

Provide a concise, accurate answer using only the provided medical context.
If you don't know the answer, say you don't know.
"""

def load_documents():
    """Load markdown documents from the specified directory."""
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.md")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {str(e)}")

def split_documents(documents: list[Document]):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: list[Document]):
    """Create and persist Chroma vector store."""
    if os.path.exists(CHROMA_PATH):
        print("Clearing existing Chroma database")

        shutil.rmtree(CHROMA_PATH)

    print(f"Creating embeddings with {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': False}
    )

    db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory=CHROMA_PATH
    )
    #db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    return db

def initialize_vector_store():
    """Initialize or load the vector store."""
    if os.path.exists(CHROMA_PATH):
        print("Loading existing Chroma database")
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        print("Creating new Chroma database")
        documents = load_documents()
        chunks = split_documents(documents)
        return create_vector_store(chunks)

def query_database(db, query_text):
    """Query the database for relevant documents."""
    try:
        results = db.similarity_search_with_relevance_scores(
            query_text,
            k=TOP_K_RESULTS
        )
        if not results or results[0][1] < SIMILARITY_THRESHOLD:
            return None, None

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
        return context_text, sources
    except Exception as e:
        raise RuntimeError(f"Database query failed: {str(e)}")
