"""
Hybrid Graph: LangGraph con AutoGen come Nodo
Workflow complesso per analisi dati con team di agenti
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import autogen
from langchain_ollama import ChatOllama
import json
import random

# ============================================
# DEFINIZIONE DELLO STATE
# ============================================

class AnalysisState(TypedDict):
    """Stato condiviso nel workflow di analisi"""
    raw_data: List[float]           # Dati grezzi da analizzare
    analysis_request: str           # Richiesta di analisi dell'utente
    prepared_data: Dict[str, Any]   # Dati preparati e formattati
    autogen_analysis: str           # Risultato dell'analisi AutoGen
    final_report: str               # Report finale formattato
    workflow_steps: List[str]       # Tracciamento dei passi eseguiti


# ============================================
# CONFIGURAZIONE COMPONENTI
# ============================================

# LLM per nodi LangGraph
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.5
)

# Config AutoGen
autogen_config = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

autogen_llm_config = {
    "config_list": autogen_config,
    "temperature": 0.7,
    "timeout": 120,
}


# ============================================
# NODO 1: DATA PREPARATION
# ============================================

def data_preparation_node(state: AnalysisState) -> AnalysisState:
    """
    🔧 NODO 1: Prepara e valida i dati per l'analisi
    
    - Valida i dati grezzi
    - Calcola statistiche di base
    - Prepara formato per AutoGen
    """
    print("\n" + "="*60)
    print("🔧 NODO 1: Data Preparation")
    print("="*60)
    
    raw_data = state["raw_data"]
    
    # Validazione
    if not raw_data or len(raw_data) == 0:
        state["prepared_data"] = {"error": "Nessun dato fornito"}
        state["workflow_steps"].append("❌ Data Preparation: ERRORE - Dati vuoti")
        return state
    
    # Calcola statistiche di base
    sorted_data = sorted(raw_data)
    n = len(raw_data)
    
    prepared = {
        "data": raw_data,
        "sorted_data": sorted_data,
        "count": n,
        "min": min(raw_data),
        "max": max(raw_data),
        "sum": sum(raw_data),
        "data_preview": f"{raw_data[:5]}..." if n > 5 else str(raw_data)
    }
    
    state["prepared_data"] = prepared
    state["workflow_steps"].append(
        f"✅ Data Preparation: {n} valori processati (range: {prepared['min']:.2f} - {prepared['max']:.2f})"
    )
    
    print(f"   📊 Dati processati: {n} valori")
    print(f"   📈 Range: {prepared['min']:.2f} - {prepared['max']:.2f}")
    
    return state


# ============================================
# NODO 2: AUTOGEN ANALYSIS (CHIAVE!)
# ============================================

def autogen_analysis_node(state: AnalysisState) -> AnalysisState:
    """
    🤖 NODO 2: Analisi tramite Team AutoGen
    
    Questo è il CUORE dell'integrazione:
    - Crea un team di agenti AutoGen specializzati
    - Passa i dati e la richiesta al team
    - Cattura e restituisce l'analisi collaborativa
    """
    print("\n" + "="*60)
    print("🤖 NODO 2: AutoGen Analysis Team")
    print("="*60)
    
    prepared_data = state["prepared_data"]
    analysis_request = state["analysis_request"]
    
    # Verifica dati
    if "error" in prepared_data:
        state["autogen_analysis"] = "Errore: dati non validi"
        state["workflow_steps"].append("❌ AutoGen Analysis: SALTATO - Dati invalidi")
        return state
    
    try:
        # ============================================
        # CREAZIONE TEAM AUTOGEN
        # ============================================
        
        print("   👥 Creazione team di analisi...")
        
        # AGENTE 1: Data Analyst
        analyst = autogen.AssistantAgent(
            name="DataAnalyst",
            llm_config=autogen_llm_config,
            system_message=f"""Sei un esperto analista di dati.
            
Hai ricevuto questi dati da analizzare:
- Numero di valori: {prepared_data['count']}
- Range: {prepared_data['min']:.2f} - {prepared_data['max']:.2f}
- Dati: {prepared_data['data_preview']}

Il tuo compito:
1. Esegui l'analisi richiesta dall'utente
2. Calcola le metriche necessarie (media, mediana, outlier, ecc.)
3. Presenta i risultati in modo chiaro e strutturato

Quando hai completato l'analisi, termina con: ANALISI_COMPLETATA"""
        )
        
        # AGENTE 2: Statistical Critic
        critic = autogen.AssistantAgent(
            name="StatisticalCritic",
            llm_config=autogen_llm_config,
            system_message="""Sei un critico statistico esperto.

Il tuo compito:
1. Verifica la correttezza dei calcoli dell'analista
2. Identifica eventuali errori metodologici
3. Suggerisci miglioramenti o approfondimenti

Sii costruttivo ma rigoroso. Termina con: REVISIONE_COMPLETATA"""
        )
        
        # AGENTE 3: User Proxy (Coordinatore)
        user_proxy = autogen.UserProxyAgent(
            name="Coordinatore",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            is_termination_msg=lambda x: "ANALISI_COMPLETATA" in x.get("content", "")
        )
        
        # ============================================
        # GROUP CHAT SETUP
        # ============================================
        
        groupchat = autogen.GroupChat(
            agents=[user_proxy, analyst, critic],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=autogen_llm_config
        )
        
        # ============================================
        # COSTRUZIONE PROMPT PER AUTOGEN
        # ============================================
        
        analysis_prompt = f"""
TASK DI ANALISI DATI:

{analysis_request}

DATI DISPONIBILI:
- Numero totale: {prepared_data['count']} valori
- Valore minimo: {prepared_data['min']:.2f}
- Valore massimo: {prepared_data['max']:.2f}
- Somma totale: {prepared_data['sum']:.2f}
- Dati completi: {prepared_data['data']}

Esegui l'analisi richiesta e presenta i risultati in modo chiaro.
Include calcoli specifici e insights rilevanti.
"""
        
        print("   🚀 Avvio conversazione AutoGen...")
        
        # ============================================
        # ESECUZIONE TEAM AUTOGEN
        # ============================================
        
        chat_result = user_proxy.initiate_chat(
            manager,
            message=analysis_prompt
        )
        
        # ============================================
        # ESTRAZIONE RISULTATI
        # ============================================
        
        # Cerca l'ultimo messaggio significativo dell'analista
        analysis_result = ""
        
        if hasattr(chat_result, 'chat_history'):
            messages = chat_result.chat_history
        else:
            messages = []
        
        # Estrai messaggi degli agenti
        for msg in reversed(messages):
            sender = msg.get('name', msg.get('role', ''))
            content = msg.get('content', '')
            
            if 'DataAnalyst' in sender and len(content) > 100:
                analysis_result = content
                break
        
        if not analysis_result:
            # Fallback: prendi l'ultimo messaggio non vuoto
            for msg in reversed(messages):
                content = msg.get('content', '')
                if len(content) > 50:
                    analysis_result = content
                    break
        
        if not analysis_result:
            analysis_result = "⚠️ AutoGen ha completato l'analisi ma non è stato possibile estrarre i risultati."
        
        state["autogen_analysis"] = analysis_result
        state["workflow_steps"].append(
            f"✅ AutoGen Analysis: Completata ({len(messages)} scambi tra agenti)"
        )
        
        print(f"   ✅ Analisi completata: {len(analysis_result)} caratteri")
        print(f"   💬 Scambi totali: {len(messages)}")
        
    except Exception as e:
        error_msg = f"❌ Errore durante analisi AutoGen: {str(e)}"
        state["autogen_analysis"] = error_msg
        state["workflow_steps"].append(f"❌ AutoGen Analysis: ERRORE - {str(e)}")
        print(f"   {error_msg}")
    
    return state


# ============================================
# NODO 3: FINAL REPORT
# ============================================

def final_report_node(state: AnalysisState) -> AnalysisState:
    """
    📊 NODO 3: Genera il report finale formattato
    
    - Combina dati preparati e analisi AutoGen
    - Crea un report strutturato e leggibile
    - Include metadata del workflow
    """
    print("\n" + "="*60)
    print("📊 NODO 3: Final Report Generation")
    print("="*60)
    
    prepared_data = state["prepared_data"]
    autogen_analysis = state["autogen_analysis"]
    analysis_request = state["analysis_request"]
    
    # Costruisci il report
    report_sections = []
    
    # Header
    report_sections.append("# 📊 REPORT DI ANALISI DATI")
    report_sections.append("=" * 60)
    report_sections.append("")
    
    # Richiesta originale
    report_sections.append("## 🎯 Richiesta di Analisi")
    report_sections.append(f"{analysis_request}")
    report_sections.append("")
    
    # Informazioni sui dati
    report_sections.append("## 📈 Informazioni sui Dati")
    if "error" not in prepared_data:
        report_sections.append(f"- **Numero di valori**: {prepared_data['count']}")
        report_sections.append(f"- **Valore minimo**: {prepared_data['min']:.2f}")
        report_sections.append(f"- **Valore massimo**: {prepared_data['max']:.2f}")
        report_sections.append(f"- **Somma totale**: {prepared_data['sum']:.2f}")
        report_sections.append("")
    
    # Analisi AutoGen
    report_sections.append("## 🤖 Analisi del Team AutoGen")
    report_sections.append(autogen_analysis)
    report_sections.append("")
    
    # Workflow steps
    report_sections.append("## 🔄 Passi del Workflow")
    for step in state["workflow_steps"]:
        report_sections.append(f"- {step}")
    
    # Combina tutto
    final_report = "\n".join(report_sections)
    
    state["final_report"] = final_report
    state["workflow_steps"].append("✅ Final Report: Generato")
    
    print("   ✅ Report generato con successo")
    print(f"   📄 Lunghezza: {len(final_report)} caratteri")
    
    return state


# ============================================
# COSTRUZIONE DEL GRAFO
# ============================================

def create_hybrid_analysis_graph():
    """
    Crea il grafo ibrido LangGraph + AutoGen
    
    Flusso:
    START → Data Preparation → AutoGen Analysis → Final Report → END
    """
    
    workflow = StateGraph(AnalysisState)
    
    # Aggiungi nodi
    workflow.add_node("prepare_data", data_preparation_node)
    workflow.add_node("autogen_analysis", autogen_analysis_node)
    workflow.add_node("final_report", final_report_node)
    
    # Definisci flusso lineare
    workflow.set_entry_point("prepare_data")
    workflow.add_edge("prepare_data", "autogen_analysis")
    workflow.add_edge("autogen_analysis", "final_report")
    workflow.add_edge("final_report", END)
    
    return workflow.compile()


# ============================================
# FUNZIONE DI ESECUZIONE
# ============================================

def run_hybrid_analysis(
    data: List[float],
    analysis_request: str
) -> Dict[str, Any]:
    """
    Esegue il workflow completo di analisi ibrida
    
    Args:
        data: Lista di valori numerici da analizzare
        analysis_request: Descrizione dell'analisi richiesta
        
    Returns:
        Dict con report finale e dettagli del workflow
    """
    
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🚀  AVVIO WORKFLOW IBRIDO LANGGRAPH + AUTOGEN")
    print("🚀" + "="*58 + "🚀")
    
    # Inizializza stato
    initial_state = {
        "raw_data": data,
        "analysis_request": analysis_request,
        "prepared_data": {},
        "autogen_analysis": "",
        "final_report": "",
        "workflow_steps": []
    }
    
    # Crea ed esegui il grafo
    graph = create_hybrid_analysis_graph()
    result = graph.invoke(initial_state)
    
    print("\n" + "✅" + "="*58 + "✅")
    print("✅  WORKFLOW COMPLETATO")
    print("✅" + "="*58 + "✅\n")
    
    return {
        "report": result["final_report"],
        "workflow_steps": result["workflow_steps"],
        "raw_data": result["raw_data"],
        "prepared_data": result["prepared_data"],
        "autogen_analysis": result["autogen_analysis"]
    }


# ============================================
# FUNZIONI DI UTILITÀ
# ============================================

def generate_sample_data(data_type: str = "normal") -> List[float]:
    """Genera dati di esempio per testing"""
    
    if data_type == "normal":
        # Dati normali con alcuni outlier
        data = [random.gauss(100, 15) for _ in range(20)]
        data.extend([150, 45])  # Aggiungi outlier
        
    elif data_type == "sales":
        # Simula dati di vendita
        data = [random.uniform(1000, 5000) for _ in range(12)]
        
    elif data_type == "temperature":
        # Simula temperature
        data = [random.uniform(15, 30) for _ in range(30)]
        
    else:
        # Default
        data = [random.uniform(0, 100) for _ in range(15)]
    
    return [round(x, 2) for x in data]