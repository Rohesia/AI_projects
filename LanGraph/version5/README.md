# 🤖 Multi-Agent AI Application
**LangGraph + AutoGen + Ollama (Local LLM)**

Un'applicazione avanzata multi-agente che integra:

- 🧠 **LangGraph** per orchestrazione e routing intelligente
- 🤖 **AutoGen** per team collaborativi di agenti
- 🦙 **Ollama (Llama3)** come LLM locale
- 📚 **RAG** (Retrieval-Augmented Generation) con ricerca semantica
- 🔬 **Workflow ibrido**: LangGraph + AutoGen nello stesso grafo

---

## 🚀 Overview

Questa applicazione dimostra tre paradigmi avanzati di AI orchestration:

1. **RAG intelligente** con Query Router
2. **Team multi-agente** collaborativi
3. **Workflow ibrido**: AutoGen come nodo dentro LangGraph

L'obiettivo è mostrare come combinare:
- Retrieval
- Routing decisionale
- Orchestrazione a grafo
- Multi-agent collaboration
- LLM locale

Il tutto in un'unica applicazione modulare e scalabile.

---

## 🏗 Architettura Generale

```
Streamlit UI
│
├── RAG LangGraph
│   ├── Query Router
│   ├── Vector Store (Chroma)
│   ├── RAG Generation
│   └── Direct Generation
│
├── AutoGen Team
│   ├── GroupChat
│   ├── Manager
│   └── Specializzati per ruolo
│
└── Hybrid Analysis
    ├── Data Preparation (LangGraph)
    ├── AutoGen Analysis Team
    └── Final Report (LangGraph)
```

**LLM utilizzato:** Ollama + llama3 (locale)

---

## 🧠 1️⃣ RAG con LangGraph (Query Router Intelligente)

### 🔍 Caratteristiche

Routing automatico tra:
- 📚 **RAG** (ricerca nei documenti)
- 🧠 **Risposta diretta**

Tecnologie:
- Vector Store con **Chroma**
- Embeddings via **Ollama**
- Visualizzazione del percorso seguito

### 🔀 Logica del Router

Il sistema analizza la query:

**Se contiene parole come:**
- "documento", "testo", "secondo", "dettagli"
→ attiva **RAG**

**Se è una domanda generale:**
- "cos'è", "spiega", "differenza tra"
→ **risposta diretta**

### 🗺 Grafo LangGraph

```
router
   ↓
retrieve ──→ rag_generation ──→ END
   │
   └────────→ direct_generation ──→ END
```

### 💡 Perché è interessante?

Non è un semplice RAG.  
È un **sistema decisionale adattivo** che sceglie il percorso migliore.

---

## 🤖 2️⃣ Team AutoGen – Agenti Collaborativi

Sistema multi-agente con **ruoli specializzati**.

### 👥 Team disponibili

#### ✍️ Creative Writing Team
- Scrittore
- Editor
- Critico

#### 🔬 Research Team
- Ricercatore
- Analista

#### 🧩 Problem Solver
- Analista
- Strategist

Ogni team usa:
- `GroupChat`
- `GroupChatManager`
- `UserProxyAgent`

### 🔄 Dinamica

```
Utente → Coordinatore
        ↓
Scrittore → Editor → Critico
        ↓
Output finale
```

### 🎯 Perché è potente?

- Simula **collaborazione reale**
- **Specializzazione** dei ruoli
- **Revisione** e validazione interna
- **Multi-turn reasoning**

---

## 🔬 3️⃣ Hybrid Analysis (LangGraph + AutoGen)

### 🌟 Il modulo più avanzato

**AutoGen viene utilizzato come nodo dentro LangGraph.**

### Workflow:

```
START
  ↓
🔧 Data Preparation (LangGraph)
  ↓
🤖 AutoGen Analysis Team
  ↓
📊 Final Report (LangGraph)
  ↓
END
```

### 🔧 Nodo 1: Data Preparation

- Validazione dati
- Calcolo statistiche base
- Normalizzazione input
- Logging workflow

### 🤖 Nodo 2: AutoGen Analysis

Team composto da:
- 📊 **DataAnalyst**
- 🔍 **StatisticalCritic**

Funzioni:
- Analisi collaborativa
- Revisione metodologica
- Verifica calcoli
- Multi-turn discussion

### 📊 Nodo 3: Final Report

LangGraph:
- Aggrega risultati
- Struttura report Markdown
- Inserisce metadata workflow
- Prepara output scaricabile

---

## 🛠 Stack Tecnologico

| Componente | Tecnologia |
|------------|------------|
| LLM | Ollama + Llama3 |
| Orchestrazione | LangGraph |
| Multi-Agent | AutoGen |
| Vector Store | Chroma |
| Embeddings | OllamaEmbeddings |
| UI | Streamlit |
| Text Splitting | RecursiveCharacterTextSplitter |

---

## 📂 Struttura del Progetto

```
.
├── home.py
├── pages/
│   ├── 1_rag_langraph.py
│   ├── 2_autogen_team.py
│   └── 3_hybrid_analysis.py
│
├── utils/
│   ├── rag_graph.py
│   ├── autogen_team.py
│   └── hybrid_graph.py
│
└── README.md
```

---

## ⚙️ Setup Locale

### 1️⃣ Installazione dipendenze

```bash
pip install -r requirements.txt
```

### 2️⃣ Installa Ollama

Scarica da: [https://ollama.ai](https://ollama.ai)

Avvia server:
```bash
ollama serve
```

### 3️⃣ Scarica Llama3

```bash
ollama pull llama3
```

### 4️⃣ Avvia l'app

```bash
streamlit run home.py
```

---

## 📊 Esempi di Utilizzo

### RAG

- *"Cosa dice il documento sul Machine Learning?"*
- *"Secondo il testo, quali applicazioni sono citate?"*

### AutoGen

- *"Scrivi un articolo sui benefici dell'AI in sanità"*
- *"Strategia per implementare un sistema RAG"*

### Hybrid

**Dati:**
```
10, 12, 15, 200, 14, 13
```

**Analisi:**
- Identifica outlier e calcola media senza outlier

**Output:**
- Analisi collaborativa
- Revisione statistica
- Report strutturato

---

## 🧩 Design Patterns Implementati

- ✅ State-based orchestration (LangGraph)
- ✅ Conditional routing
- ✅ Multi-agent coordination
- ✅ Separation of concerns
- ✅ LLM-as-a-service locale
- ✅ Workflow tracking & logging

---

## 🎯 Perché Questo Progetto è Avanzato

- ✔ Integra **2 framework complessi**
- ✔ Usa **LLM locale** (no API esterne)
- ✔ Dimostra **orchestration + collaboration**
- ✔ Mostra **reasoning multi-step**
- ✔ Ha **UI interattiva**
- ✔ Architettura **modulare**
- ✔ Separazione tra **logica e presentazione**

**È un progetto perfetto per:**
- Portfolio AI Engineer
- Colloquio AI Specialist
- Tesi magistrale
- Dimostrazione di agentic systems

---

## 🔮 Possibili Estensioni Future

- [ ] Memory persistente tra sessioni
- [ ] Tool usage per agenti (Python execution)
- [ ] Visualizzazione del grafo LangGraph
- [ ] Logging strutturato con tracing
- [ ] Dockerizzazione
- [ ] Deployment su server remoto
- [ ] Aggiunta di guardrail e validazione output

---

## 🧠 Concetti AI Dimostrati

- Retrieval-Augmented Generation
- Multi-Agent Systems
- Orchestration Graph
- Conditional Execution
- Collaborative Reasoning
- LLM Routing
- Prompt Engineering per ruoli

---

## 👨‍💻 Autore

Progetto sviluppato come dimostrazione avanzata di:
- AI Orchestration
- Agent Systems
- LLM Engineering
- Multi-framework integration

---

## ❤️ Conclusione

Questa applicazione non è solo una demo.

È un **laboratorio di sistemi agentici moderni**, che mostra come:

- **LangGraph** orchestra
- **AutoGen** collabora
- **Ollama** genera
- **Streamlit** visualizza

Un esempio concreto di **AI systems engineering moderno**.

---

## 📝 License

MIT License

---

