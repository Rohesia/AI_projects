"""
Pagina RAG con LangGraph e Query Router
"""

import streamlit as st
import sys
from pathlib import Path

# Aggiungi utils al path
sys.path.append(str(Path(__file__).parent.parent))

from utils.rag_graph import initialize_vectorstore, query_graph

# ============================================
# CONFIGURAZIONE
# ============================================

st.set_page_config(
    page_title="RAG LangGraph",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = []

# ============================================
# HEADER
# ============================================

st.title(" RAG Intelligente con LangGraph")
st.markdown("Sistema RAG con routing automatico tra ricerca documentale e conoscenza diretta")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header(" Gestione Documenti")
    
    # Testo manuale
    st.subheader("Inserisci Testo")
    manual_text = st.text_area(
        "Incolla il tuo documento:",
        height=200,
        placeholder="Il Machine Learning è una branca dell'intelligenza artificiale..."
    )
    
    # File upload
    st.subheader("Carica File")
    uploaded_file = st.file_uploader("Carica .txt", type=["txt"])
    
    # Inizializzazione
    if st.button(" Inizializza Vector Store", type="primary"):
        documents = []
        
        if manual_text.strip():
            documents.append(manual_text)
        
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            documents.append(content)
        
        if documents:
            with st.spinner("Inizializzazione..."):
                initialize_vectorstore(documents)
                st.session_state.vectorstore_ready = True
            st.success(f" {len(documents)} documento/i caricato/i!")
        else:
            st.error(" Aggiungi almeno un documento!")
    
    st.divider()
    
    # Status
    if st.session_state.vectorstore_ready:
        st.success("🟢 Vector Store Attivo")
    else:
        st.warning("🟡 Nessun Documento Caricato")
    
    # Reset
    if st.button(" Reset Chat"):
        st.session_state.rag_chat_history = []
        st.rerun()
    
    st.divider()
    
    # Info routing
    with st.expander("ℹ️ Come Funziona il Router"):
        st.markdown("""
        **Percorso RAG** 
        - Query sui documenti
        - Parole: "documento", "testo", "dice"
        
        **Percorso Diretto** 
        - Domande generali
        - Parole: "cos'è", "spiega", "come"
        """)

# ============================================
# CHAT INTERFACE
# ============================================

st.header(" Chat")

# Mostra storico
for entry in st.session_state.rag_chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        
        # Badge percorso
        if entry["route"] == "rag":
            st.info(f" {entry['path']}")
        else:
            st.success(f" {entry['path']}")

# Input
question = st.chat_input("Fai una domanda...")

if question:
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Elaborazione..."):
            result = query_graph(question)
        
        st.write(result["answer"])
        
        route = result["route_decision"]
        path = result["path_taken"]
        
        if route == "rag":
            st.info(f" **Percorso:** {path}")
        else:
            st.success(f" **Percorso:** {path}")
    
    st.session_state.rag_chat_history.append({
        "question": question,
        "answer": result["answer"],
        "route": route,
        "path": path
    })
    
    st.rerun()

# ============================================
# ESEMPI
# ============================================

st.divider()

with st.expander("💡 Esempi di Query"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Query RAG:**")
        examples_rag = [
            "Cosa dice il documento sul ML?",
            "Secondo il testo, quali applicazioni?",
            "Informazioni specifiche su..."
        ]
        for ex in examples_rag:
            if st.button(ex, key=f"rag_{ex}"):
                st.session_state.next_question = ex
                st.rerun()
    
    with col2:
        st.markdown("**Query Dirette:**")
        examples_direct = [
            "Cos'è il Machine Learning?",
            "Spiega le reti neurali",
            "Differenza tra AI e ML"
        ]
        for ex in examples_direct:
            if st.button(ex, key=f"direct_{ex}"):
                st.session_state.next_question = ex
                st.rerun()