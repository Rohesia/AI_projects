"""
Pagina Team AutoGen - Agenti Collaborativi
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.autogen_team import run_autogen_team, AVAILABLE_TEAMS

# ============================================
# CONFIGURAZIONE
# ============================================

st.set_page_config(
    page_title="AutoGen Team",
    page_icon="🤖",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================

if "autogen_history" not in st.session_state:
    st.session_state.autogen_history = []
if "current_team" not in st.session_state:
    st.session_state.current_team = "creative_writer"

# ============================================
# HEADER
# ============================================

st.title("🤖 Team AutoGen - Agenti Collaborativi")
st.markdown("Vedi gli agenti AI collaborare per risolvere compiti complessi!")

# ============================================
# SIDEBAR - CONFIGURAZIONE TEAM
# ============================================

with st.sidebar:
    st.header("⚙️ Configurazione Team")
    
    # Selezione team
    team_choice = st.selectbox(
        "Scegli il Team:",
        options=list(AVAILABLE_TEAMS.keys()),
        format_func=lambda x: AVAILABLE_TEAMS[x]["name"],
        key="team_selector"
    )
    
    st.session_state.current_team = team_choice
    
    # Info sul team selezionato
    team_info = AVAILABLE_TEAMS[team_choice]
    
    st.markdown(f"### {team_info['icon']} {team_info['name']}")
    st.markdown(f"**Descrizione:** {team_info['description']}")
    
    st.markdown("**Agenti:**")
    for agent in team_info['agents']:
        st.markdown(f"- {agent}")
    
    st.divider()
    
    # Impostazioni avanzate
    with st.expander("🔧 Impostazioni Avanzate"):
        max_rounds = st.slider(
            "Max Round Conversazione:",
            min_value=2,
            max_value=10,
            value=5,
            help="Numero massimo di scambi tra agenti"
        )
        
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Creatività delle risposte"
        )
    
    st.divider()
    
    # Reset
    if st.button(" Reset Conversazioni", type="secondary"):
        st.session_state.autogen_history = []
        st.success(" Reset completato!")
        st.rerun()
    
    # Info Ollama
    st.divider()
    st.info("""
    **Status Ollama**
    
    Verifica che sia attivo:
```bash
    ollama serve
```
    
    Modello: `llama3`
    """)

# ============================================
# MAIN CONTENT
# ============================================

# Tabs per organizzazione
tab1, tab2 = st.tabs([" Conversazione", " Statistiche"])

with tab1:
    st.header(" Chat con il Team")
    
    # Mostra storico
    for i, entry in enumerate(st.session_state.autogen_history):
        with st.expander(
            f" Conversazione {i+1}: {entry['prompt'][:50]}...",
            expanded=(i == len(st.session_state.autogen_history) - 1)
        ):
            st.markdown(f"**Team:** {entry['team_name']}")
            st.markdown(f"**Prompt:** {entry['prompt']}")
            
            st.divider()
            
            # Mostra messaggi degli agenti
            st.markdown("**💬 Scambi tra Agenti:**")
            
            for msg in entry['messages']:
                # Identifica il ruolo
                role = msg.get('role', msg.get('name', 'Unknown'))
                content = msg.get('content', '')
                
                # Salta messaggi di sistema
                if not content or content == "":
                    continue
                
                # Colore basato sul ruolo
                if 'user' in role.lower() or 'coordinatore' in role.lower():
                    icon = "👤"
                    color = "blue"
                else:
                    icon = "🤖"
                    color = "green"
                
                st.markdown(f":{color}[**{icon} {role}:**]")
                st.markdown(content)
                st.markdown("---")
            
            # Risultato finale
            if entry.get('final_result'):
                st.success("✅ **Risultato Finale:**")
                st.markdown(entry['final_result'])

    # Input utente
    st.divider()
    
    prompt_input = st.text_area(
        "Inserisci il tuo prompt per il team:",
        height=100,
        placeholder="Es: Scrivi un breve articolo sui benefici del Machine Learning per la sanità",
        key="user_prompt"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_button = st.button(
            "🚀 Esegui",
            type="primary",
            disabled=not prompt_input.strip()
        )
    
    with col2:
        if st.button("🔄 Esempio Casuale"):
            examples = team_info.get('examples', [
                "Scrivi un breve riassunto sui benefici del ML",
                "Crea una lista dei vantaggi dell'AI nella medicina"
            ])
            import random
            st.session_state.example_prompt = random.choice(examples)
            st.rerun()
    
    # Esegui team
    if run_button and prompt_input.strip():
        with st.spinner(f"🤖 Il team {team_info['name']} sta lavorando..."):
            try:
                result = run_autogen_team(
                    prompt=prompt_input,
                    team_type=team_choice,
                    max_rounds=max_rounds,
                    temperature=temperature
                )
                
                # Salva nello storico
                st.session_state.autogen_history.append({
                    "team_name": team_info['name'],
                    "prompt": prompt_input,
                    "messages": result['messages'],
                    "final_result": result.get('summary', ''),
                    "team_type": team_choice
                })
                
                st.success("✅ Team ha completato il task!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Errore: {e}")
                st.exception(e)

with tab2:
    st.header("📊 Statistiche")
    
    if st.session_state.autogen_history:
        total_convs = len(st.session_state.autogen_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conversazioni Totali", total_convs)
        
        with col2:
            total_msgs = sum(len(h['messages']) for h in st.session_state.autogen_history)
            st.metric("Messaggi Totali", total_msgs)
        
        with col3:
            avg_msgs = total_msgs / total_convs if total_convs > 0 else 0
            st.metric("Media Msg/Conv", f"{avg_msgs:.1f}")
        
        # Grafico uso team
        st.divider()
        st.subheader("Uso dei Team")
        
        team_usage = {}
        for entry in st.session_state.autogen_history:
            team = entry['team_name']
            team_usage[team] = team_usage.get(team, 0) + 1
        
        st.bar_chart(team_usage)
        
    else:
        st.info("Nessuna conversazione ancora. Inizia a chattare con il team!")

# ============================================
# ESEMPI RAPIDI
# ============================================

st.divider()

with st.expander("💡 Esempi di Prompt per Questo Team"):
    examples = team_info.get('examples', [])
    
    if examples:
        for ex in examples:
            if st.button(ex, key=f"example_{ex[:20]}"):
                st.session_state.user_prompt = ex
                st.rerun()
    else:
        st.write("Nessun esempio disponibile per questo team.")