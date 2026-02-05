# ================================================================
# L'AGGREGATORE DOCUMENTALE AVANZATO
# ================================================================
# Regole rispettate:
#   [1] Ingestion eterogenea       -> loader scelto per estensione
#   [2] Context Window Management  -> TokenTextSplitter (token-based)
#   [3] Quality Control            -> soglia similarity_threshold
#   [4] Prompt Dinamico            -> tono + lingua come parametri
#   [5] Pure LCEL                  -> niente create_stuff/create_retrieval
#
# ================================================================


import os
import sys
import shutil
from pathlib import Path
from typing import List

# --- LangChain core ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# --- Ollama ---
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- ChromaDB ---
from langchain_chroma import Chroma

# --- Loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

import os
import sys
import shutil
from pathlib import Path
from typing import List

# --- LangChain core ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# --- Ollama ---
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- ChromaDB ---
from langchain_chroma import Chroma

# --- Loaders ---
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

from langchain_text_splitters import TokenTextSplitter
DOCUMENTS_PATH   = "./data"
PERSIST_DIRECTORY = "./chroma_db"
RESET_DB         = False                 

EMBEDDING_MODEL  = "llama3"    
LLM_MODEL        = "llama3"


CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 100
SIMILARITY_THRESHOLD = 0.3
### 1. INGESTION  –  Multi-Source Loader manuale
# Mappa: estensione -> classe loader
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv":  CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls":  UnstructuredExcelLoader,
}

def detect_and_load(file_path: str) -> List[Document]:
    """
    Rileva l'estensione del file e invoca il loader corretto.
    Aggiunge metadata 'source_file' con il nome del file.
    """
    ext = Path(file_path).suffix.lower()

    if ext not in LOADER_MAP:
        print(f"  [SKIP] Formato non supportato: {file_path}")
        return []

    loader_cls = LOADER_MAP[ext]
    loader     = loader_cls(file_path)
    docs       = loader.load()

    # Normalizzazione metadata
    for doc in docs:
        doc.metadata["source_file"] = Path(file_path).name

    print(f"  [OK]   {Path(file_path).name} -> {len(docs)} documento/i caricato/i")
    return docs


def load_all_documents() -> List[Document]:
    """
    Scansiona DOCUMENTS_PATH, rileva ogni file, carica con il loader giusto.
    """
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        print(f"[INFO] Creata '{DOCUMENTS_PATH}'. Inserisci i file e rilancia.")
        return []

    print("[INFO] Scansione documenti...")
    all_docs: List[Document] = []

    for file_path in sorted(Path(DOCUMENTS_PATH).iterdir()):
        if file_path.is_file():
            docs = detect_and_load(str(file_path))
            all_docs.extend(docs)

    print(f"[INFO] Totale documenti caricati: {len(all_docs)}")
    return all_docs

### 2. SPLITTING  –  Token-based (unico ammesso)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Divide i documenti in chunk usando TokenTextSplitter.
    """
    if not documents:
        return []

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Chunk generati: {len(chunks)}")
    return chunks


### 3. VECTORSTORE  –  Sync intelligente

def reset_vectorstore():
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print("[INFO] Vectorstore resettato.")


def sync_vectorstore(chunks: List[Document], embeddings) -> Chroma:
    """
    Crea o aggiorna il vectorstore.
    Aggiunge solo i chunk ancora mancanti nel DB.
    """
    vectorstore = Chroma(
        collection_name="aggregatore_docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    if not chunks:
        print("[WARN] Nessun chunk da indicizzare.")
        return vectorstore

    # Confronto con i testi già presenti
    existing  = vectorstore.get()
    existing_texts = set(existing["documents"]) if existing["documents"] else set()
    new_chunks = [c for c in chunks if c.page_content not in existing_texts]

    if new_chunks:
        vectorstore.add_documents(documents=new_chunks)
        print(f"[INFO] Chunk aggiunti al vectorstore: {len(new_chunks)}")
    else:
        print("[INFO] Vectorstore già aggiornato.")

    return vectorstore

###  QUALITY CONTROL

def retrieve_and_filter(vectorstore: Chroma, query_text: str) -> List[Document]:
    """
    Recupera i chunk più simili e scarta quelli sotto SIMILARITY_THRESHOLD.
    Stampa anche un log dei punteggi per trasparenza.
    """
    # Recupera fino a 5 candidati con punteggio
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query=query_text,
        k=5,
    )

    # Log dei punteggi grezzi
    print("\n  [QUALITY CONTROL] Punteggi recuperati:")
    for doc, score in results_with_scores:
        src  = doc.metadata.get("source_file", "?")
        flag = "✓" if score >= SIMILARITY_THRESHOLD else "✗"
        print(f"    {flag} score={score:.3f}  [{src}]")

    # Filtra: tieni solo quelli sopra la soglia
    filtered = [doc for doc, score in results_with_scores if score >= SIMILARITY_THRESHOLD]

    print(f"  [QUALITY CONTROL] Chunk passati il filtro: {len(filtered)}/{len(results_with_scores)}\n")
    return filtered




SYSTEM_PROMPT = (
    "Sei un aggregatore documentale esperto. Il tuo compito è rispondere "
    "usando SOLO le informazioni presenti nel contesto fornito.\n"
    "Se il contesto non contiene la risposta, dilo esplicitamente.\n\n"
    "--- CONTESTO RECUPERATO ---\n"
    "{context}\n"
    "--- FINE CONTESTO ---\n\n"
    "Rispondi nel tono specificato e nella lingua richiesta.\n"
    "Tono:   {tone}\n"
    "Lingua: {lingua}\n"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Domanda: {input}"),
])
### 5. PURE LCEL
#
# Struttura della pipeline:
#
#   input (dict)
#     |
#     ├── "input"  ──────────────────────────> retriever ──> format_docs ──> "context"
#     |                                                                        |
#     ├── "tone"   ──────────────────────────────────────────────────────>     |
#     |                                                                        v
#     ├── "lingua" ──────────────────────────────────────────> prompt_template (assembla)
#     |                                                                        |
#     └── "input"  ──────────────────────────────────────────>                 v
#                                                              llm ──> AIMessage
#                                                                        |
#                                                                        v
#                                                              parse_output (stringa)
#

def format_docs(docs: List[Document]) -> str:
    """
    Formatta i Document recuperati in una stringa pulita per il prompt.
    Aggiunge anche il nome del file sorgente per trasparenza.
    """
    if not docs:
        return "[Nessun documento rilevante trovato nella knowledge base.]"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "sconosciuto")
        parts.append(f"[Documento {i} — {source}]\n{doc.page_content}")

    return "\n\n".join(parts)


def build_lcel_chain(vectorstore: Chroma, llm):
    """
    Costruisce la chain RAG in pure LCEL.
    Il recupero usa retrieve_and_filter (quality control manuale).
    """

    # Step 1: recupera i chunk con filtro e li formatta
    retrieval_step = (
        RunnableLambda(lambda x: retrieve_and_filter(vectorstore, x["input"]))
        | RunnableLambda(format_docs)
    )

    # Step 2: costruisce il dizionario che va nel prompt
    # RunnableParallel esegue in parallelo i vari branch
    chain_input = RunnableParallel(
        context  = retrieval_step,                          # recupera + formatta
        input    = RunnablePassthrough() | RunnableLambda(lambda x: x["input"]),
        tone     = RunnablePassthrough() | RunnableLambda(lambda x: x.get("tone", "professionale")),
        lingua   = RunnablePassthrough() | RunnableLambda(lambda x: x.get("lingua", "italiano")),
    )

    # Step 3: assembla prompt -> llm -> estrai testo
    generation_step = prompt_template | llm | RunnableLambda(lambda msg: msg.content)

    # Step 4: composizione finale
    full_chain = chain_input | generation_step

    return full_chain


def query(chain, question: str, tone: str = "professionale", lingua: str = "italiano") -> str:
    """
    Esegue una query sulla chain LCEL.
    """
    try:
        return chain.invoke({
            "input":  question,
            "tone":   tone,
            "lingua": lingua,
        })
    except Exception as e:
        return f"[ERRORE] {e}"


if __name__ == "__main__":

    # --- Reset opzionale ---
    if RESET_DB:
        reset_vectorstore()

    # --- Embeddings (nomic-embed-text via Ollama) ---
    print("[INFO] Inizializzazione embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # --- Caricamento multi-source ---
    documents = load_all_documents()

    # --- Chunking token-based ---
    chunks = chunk_documents(documents)

    # --- Sync vectorstore ---
    print("[INFO] Sincronizzazione vectorstore...")
    vectorstore = sync_vectorstore(chunks, embeddings)

    # --- LLM ---
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # --- Chain LCEL ---
    print("[INFO] Costruzione chain LCEL...")
    rag_chain = build_lcel_chain(vectorstore, llm)

    # ============================================================
    # QUERY DI ESEMPIO
    # ============================================================

    print("\n" + "=" * 60)
    print(" QUERY 1 – Sintesi (italiano, tono amichevole)")
    print("=" * 60)
    print(query(rag_chain,
                question="Fammi una sintesi dei documenti disponibili.",
                tone="amichevole",
                lingua="italiano"))

    print("\n" + "=" * 60)
    print(" QUERY 2 – Dettaglio tecnico (italiano, tono tecnico)")
    print("=" * 60)
    print(query(rag_chain,
                question="Quali sono i punti principali del documento nanotech1.pdf?",
                tone="tecnico",
                lingua="italiano"))

    print("\n" + "=" * 60)
    print(" QUERY 3 – Stessa domanda ma in inglese")
    print("=" * 60)
    print(query(rag_chain,
                question="Summarize the content of cell1.docx.",
                tone="professional",
                lingua="english"))

# ================================================================
# AGGREGATORE DOCUMENTALE AVANZATO – STREAMLIT + LCEL
# ================================================================

import streamlit as st
from pathlib import Path
from typing import List


# ================================================================
# HEADER STREAMLIT
# ================================================================
st.set_page_config(page_title="Aggregatore Documentale Avanzato", layout="wide")
st.title("📚 Aggregatore Documentale Avanzato")
st.markdown(
    """
    Inserisci una domanda, scegli il tono e la lingua della risposta.
    L'aggregatore cercherà tra i documenti caricati nella knowledge base.
    """
)

# ================================================================
# INPUT UTENTE
# ================================================================
user_query = st.text_area("✏️ Domanda:", height=100)
user_tone = st.selectbox("🎨 Tono della risposta:", ["professionale", "amichevole", "tecnico"])
user_lang = st.selectbox("🌐 Lingua della risposta:", ["italiano", "english", "español"])

# ================================================================
# BOTTONI
# ================================================================
if st.button("Genera Risposta"):
    with st.spinner("Caricamento e generazione risposta..."):
        # --- Reset opzionale vectorstore ---
        if RESET_DB:
            reset_vectorstore()

        # --- Embeddings ---
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # --- Caricamento documenti ---
        documents = load_all_documents()
        if not documents:
            st.warning("📂 Nessun documento trovato. Inserisci file in ./data")
        else:
            # --- Chunking ---
            chunks = chunk_documents(documents)

            # --- Sync vectorstore ---
            vectorstore = sync_vectorstore(chunks, embeddings)

            # --- LLM ---
            llm = ChatOllama(model=LLM_MODEL, temperature=0)

            # --- Chain LCEL ---
            rag_chain = build_lcel_chain(vectorstore, llm)

            # --- Query ---
            risposta = query(rag_chain, user_query, tone=user_tone, lingua=user_lang)
            st.markdown("### ✅ Risposta generata:")
            st.write(risposta)

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown("Aggregatore Documentale Avanzato – LCEL + LangChain + Ollama + ChromaDB")
