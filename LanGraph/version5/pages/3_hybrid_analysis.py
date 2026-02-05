"""
Pagina Hybrid Analysis - LangGraph + AutoGen Integration
Il workflow più avanzato: AutoGen come nodo in LangGraph
"""

import streamlit as st
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.hybrid_graph import run_hybrid_analysis, generate_sample_data

# ============================================
# CONFIGURAZIONE
# ============================================

st.set_page_config(
    page_title="Hybrid Analysis",
    page_icon="🔬",
    layout="wide"
)

# ============================================
# SESSION STATE
# ============================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ============================================
# HEADER
# ============================================

st.title("🔬 Analisi Ibrida: LangGraph + AutoGen")

st.markdown("""
### 🌟 Il Workflow più Avanzato

Questo sistema combina **LangGraph** per l'orchestrazione e **AutoGen** per l'analisi collaborativa:

**Flusso del Workflow:**
```
1. 🔧 Data Preparation (LangGraph)
   ↓
2. 🤖 AutoGen Analysis Team (Team di Agenti)
   ↓
3. 📊 Final Report (LangGraph)
```

Il **Team AutoGen** include:
- 📊 **Data Analyst**: Esegue i calcoli
- 🔍 **Statistical Critic**: Verifica la correttezza

""")

st.divider()

# ============================================
# SIDEBAR - CONFIGURAZIONE
# ============================================

with st.sidebar:
    st.header("⚙️ Configurazione Analisi")
    
    # Tipo di dati
    data_source = st.radio(
        "Sorgente Dati:",
        ["📝 Inserimento Manuale", "🎲 Genera Dati di Esempio"]
    )
    
    if data_source == "🎲 Genera Dati di Esempio":
        sample_type = st.selectbox(
            "Tipo di Dati:",
            ["normal", "sales", "temperature"],
            format_func=lambda x: {
                "normal": "📊 Distribuzione Normale (con outlier)",
                "sales": "💰 Dati di Vendita",
                "temperature": "🌡️ Temperature"
            }[x]
        )
        
        if st.button("🎲 Genera Nuovi Dati"):
            st.session_state.generated_data = generate_sample_data(sample_type)
            st.success(f"✅ Generati {len(st.session_state.generated_data)} valori!")
        
        # Mostra dati generati
        if "generated_data" in st.session_state:
            with st.expander("👀 Visualizza Dati Generati"):
                st.write(st.session_state.generated_data)
                st.caption(f"Count: {len(st.session_state.generated_data)}")
    
    else:
        # Input manuale
        manual_input = st.text_area(
            "Inserisci i dati (separati da virgola):",
            height=100,
            placeholder="10.5, 20.3, 15.7, 18.2, ...",
            help="Inserisci numeri separati da virgola"
        )
    
    st.divider()
    
    # Tipo di analisi
    st.subheader("📊 Tipo di Analisi")
    
    analysis_templates = {
        "basic_stats": "Calcola media, mediana e deviazione standard dei dati",
        "outliers": "Identifica gli outlier e calcola la media senza outlier",
        "trends": "Analizza i trend e identifica pattern nei dati",
        "custom": "Personalizzata..."
    }
    
    analysis_choice = st.selectbox(
        "Template di Analisi:",
        list(analysis_templates.keys()),
        format_func=lambda x: {
            "basic_stats": "📈 Statistiche Base",
            "outliers": "🎯 Rilevamento Outlier",
            "trends": "📊 Analisi Trend",
            "custom": "✏️ Personalizzata"
        }[x]
    )
    
    if analysis_choice == "custom":
        analysis_request = st.text_area(
            "Descrivi l'analisi:",
            height=100,
            placeholder="Es: Trova la media e identifica i 3 valori più alti..."
        )
    else:
        analysis_request = analysis_templates[analysis_choice]
        st.info(f"📝 {analysis_request}")
    
    st.divider()
    
    # Info sistema
    with st.expander("ℹ️ Info Sistema"):
        st.markdown("""
        **Componenti:**
        - LangGraph: Orchestrazione
        - AutoGen: Analisi collaborativa
        - Ollama: LLM locale (Llama3)
        
        **Nodi del Grafo:**
        1. Data Preparation
        2. AutoGen Analysis
        3. Final Report
        """)

# ============================================
# MAIN CONTENT
# ============================================

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Esegui Analisi", "📜 Storico", "📊 Visualizzazione"])

with tab1:
    st.header("🚀 Esegui Nuova Analisi")
    
    # Form di esecuzione
    with st.form("analysis_form"):
        st.markdown("### Verifica Configurazione")
        
        # Mostra configurazione corrente
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sorgente Dati", data_source.split()[1])
            
            if data_source == "🎲 Genera Dati di Esempio":
                if "generated_data" in st.session_state:
                    st.metric("Numero Valori", len(st.session_state.generated_data))
                else:
                    st.warning("⚠️ Genera prima i dati!")
            else:
                if manual_input:
                    try:
                        parsed = [float(x.strip()) for x in manual_input.split(",")]
                        st.metric("Numero Valori", len(parsed))
                    except:
                        st.error("❌ Formato dati non valido!")
                else:
                    st.warning("⚠️ Inserisci i dati!")
        
        with col2:
            st.metric("Tipo Analisi", 
                     analysis_choice.replace("_", " ").title())
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Avvia Workflow Ibrido",
            type="primary",
            use_container_width=True
        )
    
    # Esecuzione
    if submitted:
        # Prepara dati
        try:
            if data_source == "🎲 Genera Dati di Esempio":
                if "generated_data" not in st.session_state:
                    st.error("❌ Genera prima i dati di esempio!")
                    st.stop()
                data = st.session_state.generated_data
            else:
                if not manual_input:
                    st.error("❌ Inserisci i dati!")
                    st.stop()
                data = [float(x.strip()) for x in manual_input.split(",")]
            
            if not analysis_request:
                st.error("❌ Specifica l'analisi da eseguire!")
                st.stop()
            
        except Exception as e:
            st.error(f"❌ Errore nella preparazione dei dati: {e}")
            st.stop()
        
        # Esegui workflow
        st.markdown("---")
        st.subheader("⚙️ Esecuzione Workflow")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔧 Preparazione dati...")
        progress_bar.progress(10)
        
        with st.spinner("Esecuzione in corso..."):
            try:
                # ESECUZIONE DEL WORKFLOW IBRIDO
                result = run_hybrid_analysis(
                    data=data,
                    analysis_request=analysis_request
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Workflow completato!")
                
                # Salva nello storico
                st.session_state.analysis_history.append({
                    "data": data,
                    "request": analysis_request,
                    "result": result,
                    "timestamp": st.session_state.get("timestamp", "Now")
                })
                
                # Mostra risultati
                st.success("✅ Analisi Completata!")
                
                # Workflow steps
                with st.expander("🔄 Passi del Workflow", expanded=True):
                    for step in result["workflow_steps"]:
                        if "✅" in step:
                            st.success(step)
                        elif "❌" in step:
                            st.error(step)
                        else:
                            st.info(step)
                
                # Report finale
                st.markdown("---")
                st.markdown("### 📊 Report Finale")
                st.markdown(result["report"])
                
                # Download button
                st.download_button(
                    label="📥 Scarica Report",
                    data=result["report"],
                    file_name="analysis_report.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("")
                st.error(f"❌ Errore durante l'esecuzione: {e}")
                st.exception(e)

with tab2:
    st.header("📜 Storico Analisi")
    
    if st.session_state.analysis_history:
        for i, entry in enumerate(reversed(st.session_state.analysis_history), 1):
            with st.expander(f"Analisi #{len(st.session_state.analysis_history) - i + 1}: {entry['request'][:50]}..."):
                
                st.markdown(f"**Richiesta:** {entry['request']}")
                st.markdown(f"**Numero Dati:** {len(entry['data'])}")
                
                st.divider()
                
                st.markdown("**Report:**")
                st.markdown(entry['result']['report'])
                
                st.divider()
                
                st.markdown("**Workflow Steps:**")
                for step in entry['result']['workflow_steps']:
                    st.text(step)
        
        # Clear history
        if st.button("🗑️ Cancella Storico"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("Nessuna analisi nello storico. Esegui la prima analisi!")

with tab3:
    st.header("📊 Visualizzazione Dati")
    
    if st.session_state.analysis_history:
        latest = st.session_state.analysis_history[-1]
        
        st.subheader("Ultima Analisi")
        
        # Chart dei dati
        st.line_chart(latest['data'])
        
        # Statistiche
        col1, col2, col3, col4 = st.columns(4)
        
        data = latest['data']
        with col1:
            st.metric("Min", f"{min(data):.2f}")
        with col2:
            st.metric("Max", f"{max(data):.2f}")
        with col3:
            st.metric("Media", f"{sum(data)/len(data):.2f}")
        with col4:
            st.metric("Count", len(data))
    else:
        st.info("Esegui un'analisi per vedere le visualizzazioni!")

# ============================================
# FOOTER
# ============================================

st.divider()

st.markdown("""
### 🎯 Come Funziona

1. **LangGraph** orchestra il workflow generale
2. **Nodo AutoGen** crea un team di agenti che collaborano
3. Gli agenti AutoGen analizzano i dati in modo collaborativo
4. **LangGraph** formatta il risultato finale

Questa è la **massima integrazione** tra i due framework! 🚀
""")