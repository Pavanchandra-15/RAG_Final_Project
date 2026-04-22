# RAG-Based Customer Support Assistant  
### Built with LangGraph + Human-in-the-Loop (HITL)

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based Customer Support Assistant that answers user queries using a PDF knowledge base.

Unlike traditional chatbots or standalone LLMs, this system:
- Retrieves relevant information from documents
- Generates context-aware responses
- Avoids hallucinations
- Escalates complex queries to a human agent (HITL)

---

## Objectives

- Build an end-to-end RAG pipeline  
- Implement document ingestion and embedding storage  
- Perform semantic retrieval using vector search  
- Use LLMs for contextual answer generation  
- Design a graph-based workflow using **LangGraph**  
- Implement **Human-in-the-Loop (HITL)** escalation  
- Apply real-world system design principles  

---

## System Architecture

The system is composed of the following components:
```pre
User Query (CLI)
│
▼
LangGraph Workflow Engine
│
▼
Retrieval Layer (ChromaDB)
│
▼
LLM (LLaMA 3.1 via Groq)
│
▼
Routing Logic
┌───────────────┐
│ │
▼ ▼
Answer Node Escalation Node (HITL)
```


---

## Tech Stack

| Component            | Technology Used |
|--------------------|----------------|
| Language           | Python         |
| Framework          | LangChain, LangGraph |
| Embeddings         | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database    | ChromaDB       |
| LLM                | LLaMA 3.1 (Groq API) |
| PDF Processing     | PyPDFLoader    |
| Interface          | CLI (Terminal) |

---

## Workflow Explanation

### 1. Ingestion Pipeline (Offline)
- Load PDF document  
- Split into chunks (1000 chars, 200 overlap)  
- Generate embeddings  
- Store in ChromaDB  

---

### 2. Query Pipeline (Online)
1. User inputs query  
2. Query passed to LangGraph  
3. Retrieve top-3 relevant chunks  
4. Generate answer using LLM  
5. Apply routing logic  

---

### 3. Conditional Routing
- If answer is confident → return to user  
- If low confidence → escalate to human  

---

### 4. Human-in-the-Loop (HITL)
- Displays query and bot response  
- Human agent provides final answer  
- Logged for auditing  

---

## Project Structure
```pre
├──src
    ├── ingest.py # PDF processing, chunking, embeddings
    ├── retriever.py # Vector retrieval (top-k search)
    ├── llm.py # LLM interaction & prompt handling
    ├── graph.py # LangGraph workflow definition
    ├── hitl.py # Human-in-the-loop module
    ├── main.py # CLI interface (entry point)
├── chroma_db/ # Vector database storage
├── data/
│ └── knowledge_base.pdf
├── escalation_log.txt
└── README.md
```


---

## Key Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Context-aware responses
- ✅ Reduced hallucination
- ✅ Graph-based workflow (LangGraph)
- ✅ Conditional routing logic
- ✅ Human fallback (HITL)
- ✅ Modular architecture

---

## Design Decisions

- **Chunk Size (1000 + overlap 200)**  
  Balances context retention and retrieval precision  

- **Top-K Retrieval (k=3)**  
  Provides sufficient context without increasing cost  

- **Phrase-based Routing**  
  Avoids extra LLM calls for confidence scoring  

- **Local Embeddings**  
  Eliminates dependency on paid APIs  

---

## Limitations

- Single PDF support  
- CLI-based interface (no UI)  
- Synchronous HITL (manual input required)  
- Phrase-based routing may miss edge cases  

---

## Future Enhancements

- Multi-document support  
- Web interface (React / Streamlit)  
- Async HITL (Slack / Email integration)  
- Conversation memory  
- LLM-based confidence scoring  
- Deployment using FastAPI  

---

## Testing Strategy

- Unit testing for each module:
  - ingestion  
  - retrieval  
  - LLM  
  - graph flow  
- End-to-end testing via CLI  

### Sample Queries

| Query | Expected Behavior |
|------|-----------------|
| What is this document about? | Answer from PDF |
| What are submission guidelines? | Retrieve and summarize |
| CEO phone number? | Escalate |
| Weather today? | Escalate |

---

## Scalability Considerations

- Replace ChromaDB with Pinecone for distributed storage  
- Use FastAPI for concurrent users  
- Implement caching to reduce latency  
- Async workflow execution  

---

## Conclusion

This project demonstrates the practical implementation of a **RAG-based intelligent system** that combines retrieval, generation, and human validation.

It highlights how modern AI systems can be designed to be:
- Accurate  
- Reliable  
- Scalable  
- Human-aware  

---

## Author

**Pavanchandra Devang L**  
RAG Internship Project – Innomatics Research Labs  

---
