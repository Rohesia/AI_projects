#Versione 2: V1+  Introduzione di AutoGen (Esercizio Base)

import autogen
from autogen import AssistantAgent, UserProxyAgent

# ============================================
# CONFIGURAZIONE LLAMA3 LOCALE VIA OLLAMA
# ============================================

config_list = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  
        "api_type": "openai",  
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# ============================================
# AGENTI DEL TEAM
# ============================================

# AGENTE 1: Il Creatore di Contenuti
creatore = AssistantAgent(
    name="CreatoreML",
    llm_config=llm_config,
    system_message="""Sei un divulgatore scientifico appassionato di Machine Learning.
    
    Scrivi un riassunto chiaro e coinvolgente (4-5 frasi) sui benefici del Machine Learning.
    Includi almeno DUE esempi concreti di applicazioni reali.
    
    Quando hai completato il tuo riassunto, termina con: FATTO"""
)

#  AGENTE 2: Il Proxy Utente
utente = UserProxyAgent(
    name="Coordinatore",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    is_termination_msg=lambda x: x.get("content", "").find("FATTO") != -1,
    code_execution_config=False,
)

# ============================================
# ESECUZIONE
# ============================================


print(" TEAM AUTOGEN: Generazione Riassunto su Machine Learning")


# Avvia la conversazione
try:
    chat_result = utente.initiate_chat(
        creatore,
        message="""Crea un riassunto coinvolgente sui benefici del Machine Learning.
        
        Requisiti:
        - Massimo 5 frasi
        - Almeno 2 esempi pratici (es. medicina, auto, ecc.)
        - Stile accessibile per non esperti
        
        Quando finisci, scrivi FATTO.""",
        max_turns=2
    )
    
    
    
except Exception as e:
    print(f"\n Errore durante l'esecuzione: {e}")
    print("\nVerifica che Ollama sia in esecuzione con: ollama serve")
    
    
