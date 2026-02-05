"""
Streamlit App per RAG con Routing Intelligente
"""

import streamlit as st
from rag_graph import initialize_vectorstore, query_graph

# ============================================
# CONFIGURAZIONE PAGINA
# ============================================

st.set_page_config(
    page_title="RAG Intelligente con LangGraph",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# INIZIALIZZAZIONE SESSION STATE
# ============================================

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================
# INTERFACCIA PRINCIPALE
# ============================================

st.title("RAG Intelligente con Routing Automatico")
st.markdown("""
**Sistema RAG avanzato** che decide automaticamente se cercare nei documenti o rispondere direttamente.
""")

# ============================================
# SIDEBAR - CARICAMENTO DOCUMENTI
# ============================================

with st.sidebar:
    st.header("Gestione Documenti")
    
    # Opzione 1: Testo manuale
    st.subheader("Opzione 1: Inserisci Testo")
    manual_text = st.text_area(
        "Incolla il tuo documento qui:",
        height=200,
        placeholder="Es: Il Machine Learning è una branca dell'AI..."
    )
    
    # Opzione 2: File upload
    st.subheader("Opzione 2: Carica File")
    uploaded_file = st.file_uploader(
        "Carica un file .txt",
        type=["txt"]
    )
    
    # Pulsante per inizializzare
    if st.button(" Inizializza Vector Store", type="primary"):
        documents = []
        
        # Raccogli documenti
        if manual_text.strip():
            documents.append(manual_text)
        
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            documents.append(content)
        
        if documents:
            with st.spinner("Inizializzazione in corso..."):
                initialize_vectorstore(documents)
                st.session_state.vectorstore_ready = True
            st.success(f" {len(documents)} documento/i caricato/i!")
        else:
            st.error("Aggiungi almeno un documento!")
    
    # Status
    st.divider()
    if st.session_state.vectorstore_ready:
        st.success("🟢 Vector Store Attivo")
    else:
        st.warning("🟡 Vector Store Non Inizializzato")
    
    # Info
    st.divider()
    st.markdown("""
    ### Come Funziona il Router
    
    **Percorso RAG** (📚):
    - Query su documenti specifici
    - Richieste di informazioni fattuali
    - Parole chiave: "documento", "secondo", "dice"
    
    **Percorso Diretto** (🧠):
    - Domande concettuali
    - Richieste di spiegazioni generali
    - Parole chiave: "cos'è", "come funziona"
    """)

# ============================================
# AREA PRINCIPALE - CHAT
# ============================================

st.header("Conversazione")

# Mostra storico chat
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        
        # Badge del percorso
        if entry["route"] == "rag":
            st.info(f"📚 {entry['path']}")
        else:
            st.success(f"{entry['path']}")

# Input utente
question = st.chat_input("Fai una domanda...")

if question:
    # Mostra domanda
    with st.chat_message("user"):
        st.write(question)
    
    # Genera risposta
    with st.chat_message("assistant"):
        with st.spinner("Elaborazione..."):
            result = query_graph(question)
        
        # Mostra risposta
        st.write(result["answer"])
        
        # Badge del percorso con icona
        route = result["route_decision"]
        path = result["path_taken"]
        
        if route == "rag":
            st.info(f"**Percorso seguito:** {path}")
        else:
            st.success(f" **Percorso seguito:** {path}")
    
    # Salva nella storia
    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"],
        "route": route,
        "path": path
    })
    
    st.rerun()

# ============================================
# ESEMPI DI QUERY
# ============================================

st.divider()

st.subheader(" Esempi di Query")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Query RAG** (cercano nei documenti):")
    st.code("""
- "Cosa dice il documento sul ML?"
- "Secondo il testo, quali sono i benefici?"
- "Informazioni specifiche su..."
    """)

with col2:
    st.markdown("**Query Dirette** (conoscenza generale):")
    st.code("""
- "Cos'è il Machine Learning?"
- "Spiega come funziona una rete neurale"
- "Differenza tra AI e ML?"
    """)