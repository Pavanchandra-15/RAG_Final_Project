import os
import sys
from typing import TypedDict

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from retriever import retrieve_relevant_chunks, format_context
from llm import generate_answer

load_dotenv()

# ── Escalation trigger phrases ─────────────────────────────────────────
ESCALATION_PHRASES = [
    "i don't have enough information",
    "let me escalate",
    "cannot answer",
    "not in the context",
    "i'm not sure",
    "unclear",
]


# ── State: the data object that flows between nodes ────────────────────
class GraphState(TypedDict):
    question:       str
    context:        str
    answer:         str
    needs_escalation: bool


# ── Node 1: Process the query ──────────────────────────────────────────
def process_node(state: GraphState) -> GraphState:
    """Retrieve relevant chunks and generate an answer."""
    print("\n[Node 1] Processing query...")
    question = state["question"]

    # Retrieve relevant chunks from ChromaDB
    chunks  = retrieve_relevant_chunks(question)
    context = format_context(chunks)

    # Generate answer using Groq LLM
    answer = generate_answer(context, question)

    print(f"[Node 1] Answer generated.")
    return {
        **state,
        "context": context,
        "answer":  answer,
    }


# ── Router: decide which path to take ─────────────────────────────────
def route_query(state: GraphState) -> str:
    """
    Check the answer for escalation phrases.
    Returns 'escalate' or 'answer' to direct the graph.
    """
    answer_lower = state["answer"].lower()
    for phrase in ESCALATION_PHRASES:
        if phrase in answer_lower:
            print("[Router] Low confidence detected → escalating to human.")
            return "escalate"
    print("[Router] Confident answer → returning to user.")
    return "answer"


# ── Node 2A: Answer node ───────────────────────────────────────────────
def answer_node(state: GraphState) -> GraphState:
    """Pass the answer through to the output."""
    return {**state, "needs_escalation": False}


# ── Node 2B: Escalation node ───────────────────────────────────────────
def escalate_node(state: GraphState) -> GraphState:
    """Mark the query for human escalation."""
    print("[Node 2] Escalating to human agent...")
    return {
        **state,
        "needs_escalation": True,
        "answer": state["answer"],
    }


# ── Build the graph ────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("process",  process_node)
    graph.add_node("answer_node",   answer_node)
    graph.add_node("escalate_node", escalate_node)

    # Set entry point
    graph.set_entry_point("process")

    # Add conditional routing after process node
    graph.add_conditional_edges(
        "process",
        route_query,
        {
            "answer":   "answer_node",
            "escalate": "escalate_node",
        }
    )

    # Both paths end the graph
    graph.add_edge("answer_node",   END)
    graph.add_edge("escalate_node", END)

    return graph.compile()


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # Test 1: Query that should be answerable
    print("=" * 50)
    print("TEST 1: Answerable query")
    print("=" * 50)
    result = app.invoke({
        "question":        "What is this document about?",
        "context":         "",
        "answer":          "",
        "needs_escalation": False,
    })
    print(f"\nFinal Answer: {result['answer']}")
    print(f"Escalated: {result['needs_escalation']}")

    # Test 2: Query that should trigger escalation
    print("\n" + "=" * 50)
    print("TEST 2: Unanswerable query (should escalate)")
    print("=" * 50)
    result2 = app.invoke({
        "question":        "What is the CEO's personal phone number?",
        "context":         "",
        "answer":          "",
        "needs_escalation": False,
    })
    print(f"\nFinal Answer: {result2['answer']}")
    print(f"Escalated: {result2['needs_escalation']}")