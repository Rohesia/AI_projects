"""
AutoGen Team Logic
Definisce diversi team di agenti specializzati
"""

import autogen
from typing import Dict, List

# ============================================
# CONFIGURAZIONE BASE
# ============================================

config_list = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]

def get_llm_config(temperature=0.7):
    """Restituisce configurazione LLM"""
    return {
        "config_list": config_list,
        "temperature": temperature,
        "timeout": 120,
    }

# ============================================
# DEFINIZIONE TEAM
# ============================================

AVAILABLE_TEAMS = {
    "creative_writer": {
        "name": "Team Scrittura Creativa",
        "icon": "✍️",
        "description": "Scrittore, Editor e Critico collaborano per contenuti di qualità",
        "agents": ["Scrittore", "Editor", "Critico"],
        "examples": [
            "Scrivi un articolo sui benefici del ML nella sanità",
            "Crea una breve guida su come iniziare con l'AI",
            "Scrivi un post blog sulle auto a guida autonoma"
        ]
    },
    "research_team": {
        "name": "Team di Ricerca",
        "icon": "🔬",
        "description": "Ricercatore e Analista per analisi approfondite",
        "agents": ["Ricercatore", "Analista", "Revisore"],
        "examples": [
            "Analizza i pro e contro del deep learning",
            "Confronta diversi approcci al NLP",
            "Ricerca sulle applicazioni dell'AI in finanza"
        ]
    },
    "problem_solver": {
        "name": "Team Problem Solving",
        "icon": "🧩",
        "description": "Analista, Strategist e Validator per risolvere problemi",
        "agents": ["Analista", "Strategist", "Validator"],
        "examples": [
            "Come migliorare l'efficienza di un algoritmo ML?",
            "Strategia per implementare un sistema RAG",
            "Ottimizzazione di pipeline di dati"
        ]
    }
}

# ============================================
# CREAZIONE TEAM
# ============================================

def create_creative_writing_team(llm_config, max_rounds=5):
    """Crea il team di scrittura creativa"""
    
    # Scrittore
    scrittore = autogen.AssistantAgent(
        name="Scrittore",
        llm_config=llm_config,
        system_message="""Sei uno scrittore creativo e competente.
        Scrivi contenuti chiari, coinvolgenti e ben strutturati.
        Quando hai finito, termina con: COMPLETATO"""
    )
    
    # Editor
    editor = autogen.AssistantAgent(
        name="Editor",
        llm_config=llm_config,
        system_message="""Sei un editor esperto.
        Rivedi il testo dello scrittore e suggerisci miglioramenti specifici.
        Concentrati su chiarezza, struttura e impatto.
        Sii costruttivo. Termina con: REVISIONATO"""
    )
    
    # Critico
    critico = autogen.AssistantAgent(
        name="Critico",
        llm_config=llm_config,
        system_message="""Sei un critico attento.
        Valuta il contenuto finale e dai un giudizio onesto.
        Evidenzia punti di forza e eventuali debolezze.
        Termina con: VALUTATO"""
    )
    
    # Coordinatore
    user_proxy = autogen.UserProxyAgent(
        name="Coordinatore",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    
    # Group Chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, scrittore, editor, critico],
        messages=[],
        max_round=max_rounds
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    return user_proxy, manager

def create_research_team(llm_config, max_rounds=5):
    """Crea il team di ricerca"""
    
    ricercatore = autogen.AssistantAgent(
        name="Ricercatore",
        llm_config=llm_config,
        system_message="""Sei un ricercatore meticoloso.
        Esplora il topic in profondità e presenta findings chiari.
        Termina con: RICERCA_COMPLETATA"""
    )
    
    analista = autogen.AssistantAgent(
        name="Analista",
        llm_config=llm_config,
        system_message="""Sei un analista critico.
        Analizza i findings del ricercatore e identifica pattern/insights.
        Termina con: ANALISI_COMPLETATA"""
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="Coordinatore",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    
    groupchat = autogen.GroupChat(
        agents=[user_proxy, ricercatore, analista],
        messages=[],
        max_round=max_rounds
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    return user_proxy, manager

def create_problem_solver_team(llm_config, max_rounds=5):
    """Crea il team problem solving"""
    
    analista = autogen.AssistantAgent(
        name="Analista",
        llm_config=llm_config,
        system_message="""Sei un analista di problemi.
        Scomponi il problema e identifica i componenti chiave.
        Termina con: ANALISI_PROBLEMA_COMPLETATA"""
    )
    
    strategist = autogen.AssistantAgent(
        name="Strategist",
        llm_config=llm_config,
        system_message="""Sei uno strategist.
        Proponi soluzioni concrete basate sull'analisi.
        Sii pratico e specifico.
        Termina con: STRATEGIA_PROPOSTA"""
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="Coordinatore",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    
    groupchat = autogen.GroupChat(
        agents=[user_proxy, analista, strategist],
        messages=[],
        max_round=max_rounds
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    return user_proxy, manager

# ============================================
# FUNZIONE PRINCIPALE
# ============================================

def run_autogen_team(
    prompt: str,
    team_type: str = "creative_writer",
    max_rounds: int = 5,
    temperature: float = 0.7
) -> Dict:
    """
    Esegue un team AutoGen con il prompt fornito
    
    Args:
        prompt: Il task da completare
        team_type: Tipo di team da usare
        max_rounds: Numero massimo di round
        temperature: Temperatura del modello
        
    Returns:
        Dict con messages e summary
    """
    
    llm_config = get_llm_config(temperature)
    
    # Seleziona il team
    if team_type == "creative_writer":
        user_proxy, manager = create_creative_writing_team(llm_config, max_rounds)
    elif team_type == "research_team":
        user_proxy, manager = create_research_team(llm_config, max_rounds)
    elif team_type == "problem_solver":
        user_proxy, manager = create_problem_solver_team(llm_config, max_rounds)
    else:
        raise ValueError(f"Team type {team_type} non riconosciuto")
    
    # Esegui la conversazione
    chat_result = user_proxy.initiate_chat(
        manager,
        message=prompt
    )
    
    # Estrai messaggi
    messages = chat_result.chat_history if hasattr(chat_result, 'chat_history') else []
    
    # Trova il risultato finale (ultimo messaggio significativo)
    final_result = ""
    for msg in reversed(messages):
        content = msg.get('content', '')
        if content and len(content) > 50:
            final_result = content
            break
    
    return {
        "messages": messages,
        "summary": final_result,
        "team_type": team_type
    }