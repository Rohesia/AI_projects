"""
Home Page - Multi-Agent Application
Navigazione tra RAG LangGraph e AutoGen Team
"""

import streamlit as st

st.set_page_config(
    page_title="Multi-Agent AI App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# HEADER
# ============================================

st.title("Multi-Agent AI Application")
st.markdown("---")

st.markdown("""
## Benvenuto nell'App Multi-Agente! 👋

Questa applicazione combina due potenti framework per l'AI:

- **LangGraph**: Per sistemi RAG intelligenti con routing
- **AutoGen**: Per team di agenti collaborativi

Scegli una pagina dalla sidebar per iniziare! 👈
""")

# ============================================
# CARDS DELLE FUNZIONALITÀ
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ###  RAG con LangGraph
    
    **Caratteristiche:**
    -  Query router intelligente
    -  Ricerca semantica nei documenti
    -  Risposta diretta o basata su RAG
    -  Visualizzazione del percorso seguito
    
    **Usa questa pagina per:**
    - Interrogare documenti specifici
    - Ottenere risposte contestualizzate
    - Vedere come il sistema decide il percorso
    """)
    
    if st.button("Vai al RAG LangGraph", type="primary", use_container_width=True):
        st.switch_page("pages/1_rag_lang.py")

with col2:
    st.markdown("""
    ### 🤖 Team AutoGen
    
    **Caratteristiche:**
    -  Multi-agente collaborativo
    -  Conversazioni intelligenti
    -  Ruoli specializzati (Scrittore, Editor, Critico)
    -  Output creativi e strutturati
    
    **Usa questa pagina per:**
    - Brainstorming creativo
    - Generazione di contenuti
    - Vedere agenti che collaborano
    """)
    
    if st.button("🚀 Vai al Team AutoGen", type="primary", use_container_width=True):
        st.switch_page("pages/2_autogen_team.py")

# ============================================
# REQUISITI TECNICI
# ============================================

st.markdown("---")

st.subheader("⚙️ Requisiti Tecnici")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Ollama**
    
    Assicurati che Ollama sia in esecuzione:
```bash
    ollama serve
```
    """)

with col2:
    st.info("""
    **Modello Llama3**
    
    Scarica il modello:
```bash
    ollama pull llama3
```
    """)

with col3:
    st.info("""
    **Porta**
    
    Default: `localhost:11434`
    
    Verifica con:
```bash
    curl localhost:11434
```
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Made with ❤️ using Streamlit, LangGraph & AutoGen
</div>
""", unsafe_allow_html=True)