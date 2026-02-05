"""
RAG Graph con Query Router Intelligente
Decide automaticamente se usare RAG o risposta diretta
"""

from typing import TypedDict, Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
import re

# ============================================
# DEFINIZIONE DELLO STATE
# ============================================

class GraphState(TypedDict):
    """Stato condiviso tra i nodi del grafo"""
    question: str              # Domanda dell'utente
    route_decision: str        # Decisione del router
    retrieved_docs: list       # Documenti recuperati
    generation: str            # Risposta finale
    path_taken: str           # Percorso seguito (per visualizzazione)


# ============================================
# INIZIALIZZAZIONE COMPONENTI
# ============================================

# Modello LLM locale
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Embeddings per la ricerca semantica
embeddings = OllamaEmbeddings(
    model="llama3",
    base_url="http://localhost:11434"
)

# Vector store globale (verrà popolato dall'app)
vectorstore = None


# ============================================
# FUNZIONI DEI NODI
# ============================================

def query_router(state: GraphState) -> GraphState:
    """
     NODO ROUTER: Decide se la query necessita di RAG o meno
    
    Criteri per attivare RAG:
    - Query su informazioni specifiche/fattuali
    - Parole chiave: "documento", "testo", "secondo", "nel documento"
    - Query che richiedono dati precisi
    """
    question = state["question"].lower()
    
    # Parole chiave che indicano necessità di RAG
    rag_keywords = [
        "documento", "testo", "secondo", "nel documento",
        "dice", "scritto", "contenuto", "informazione su",
        "dettagli", "specifico", "dati", "fonte"
    ]
    
    # Parole chiave per risposte dirette
    direct_keywords = [
        "cosa è", "cos'è", "definisci", "spiega",
        "come funziona", "perché", "quando",
        "differenza tra", "vantaggi", "svantaggi"
    ]
    
    # Logica di decisione
    needs_rag = any(keyword in question for keyword in rag_keywords)
    is_general = any(keyword in question for keyword in direct_keywords)
    
    if needs_rag or (not is_general and vectorstore is not None):
        decision = "rag"
        path = "RAG (Ricerca nei documenti)"
    else:
        decision = "direct"
        path = " Risposta Diretta (Conoscenza interna)"
    
    state["route_decision"] = decision
    state["path_taken"] = path
    
    print(f" Router Decision: {decision.upper()} - {path}")
    
    return state


def retrieve_documents(state: GraphState) -> GraphState:
    """
    NODO RAG: Recupera documenti rilevanti dal vector store
    """
    question = state["question"]
    
    if vectorstore is None:
        state["retrieved_docs"] = []
        state["path_taken"] += " (Nessun documento caricato)"
        return state
    
    # Ricerca semantica
    try:
        docs = vectorstore.similarity_search(question, k=3)
        state["retrieved_docs"] = docs
        
        print(f"📚 Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            print(f"   Doc {i}: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"❌ Errore nel recupero: {e}")
        state["retrieved_docs"] = []
    
    return state


def generate_with_rag(state: GraphState) -> GraphState:
    """
    NODO GENERAZIONE RAG: Genera risposta basata sui documenti
    """
    question = state["question"]
    docs = state["retrieved_docs"]
    
    if not docs:
        state["generation"] = "⚠️ Nessun documento rilevante trovato. Carica dei documenti prima."
        return state
    
    # Costruisci il contesto dai documenti
    context = "\n\n".join([f"Documento {i+1}:\n{doc.page_content}" 
                           for i, doc in enumerate(docs)])
    
    # Prompt per RAG
    prompt = f"""Rispondi alla seguente domanda basandoti ESCLUSIVAMENTE sui documenti forniti.
Se la risposta non è nei documenti, dillo chiaramente.

DOCUMENTI:
{context}

DOMANDA: {question}

RISPOSTA (cita i documenti quando possibile):"""
    
    try:
        response = llm.invoke(prompt)
        state["generation"] = response.content
        print(f"✅ Generazione RAG completata ({len(response.content)} chars)")
    except Exception as e:
        state["generation"] = f"❌ Errore nella generazione: {e}"
    
    return state


def generate_direct(state: GraphState) -> GraphState:
    """
     NODO GENERAZIONE DIRETTA: Risposta basata solo sulla conoscenza del modello
    """
    question = state["question"]
    
    prompt = f"""Rispondi in modo chiaro e conciso alla seguente domanda.
Usa la tua conoscenza generale. Sii breve (massimo 4-5 frasi).

DOMANDA: {question}

RISPOSTA:"""
    
    try:
        response = llm.invoke(prompt)
        state["generation"] = response.content
        print(f"✅ Generazione diretta completata ({len(response.content)} chars)")
    except Exception as e:
        state["generation"] = f"❌ Errore nella generazione: {e}"
    
    return state


def route_decision(state: GraphState) -> Literal["retrieve", "direct_generation"]:
    """
     FUNZIONE CONDIZIONALE: Decide il prossimo nodo basandosi sul router
    """
    decision = state["route_decision"]
    return "retrieve" if decision == "rag" else "direct_generation"


# ============================================
# COSTRUZIONE DEL GRAFO
# ============================================

def create_rag_graph():
    """Crea il grafo LangGraph con routing intelligente"""
    
    workflow = StateGraph(GraphState)
    
    # Aggiungi nodi
    workflow.add_node("router", query_router)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rag_generation", generate_with_rag)
    workflow.add_node("direct_generation", generate_direct)
    
    # Definisci il flusso
    workflow.set_entry_point("router")
    
    # Routing condizionale
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "direct_generation": "direct_generation"
        }
    )
    
    # Percorso RAG
    workflow.add_edge("retrieve", "rag_generation")
    workflow.add_edge("rag_generation", END)
    
    # Percorso diretto
    workflow.add_edge("direct_generation", END)
    
    return workflow.compile()


# ============================================
# FUNZIONI DI UTILITÀ
# ============================================

def initialize_vectorstore(documents: list[str]) -> Chroma:
    """Inizializza il vector store con i documenti forniti"""
    global vectorstore
    
    if not documents:
        vectorstore = None
        return None
    
    # Crea oggetti Document
    docs = [Document(page_content=doc) for doc in documents]
    
    # Split dei documenti
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    
    # Crea vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_collection"
    )
    
    print(f"✅ Vector store inizializzato con {len(splits)} chunks")
    return vectorstore


def query_graph(question: str) -> dict:
    """
    Esegue una query sul grafo e restituisce il risultato
    
    Returns:
        dict con chiavi: answer, path_taken, route_decision
    """
    graph = create_rag_graph()
    
    initial_state = {
        "question": question,
        "route_decision": "",
        "retrieved_docs": [],
        "generation": "",
        "path_taken": ""
    }
    
    result = graph.invoke(initial_state)
    
    return {
        "answer": result["generation"],
        "path_taken": result["path_taken"],
        "route_decision": result["route_decision"]
    }