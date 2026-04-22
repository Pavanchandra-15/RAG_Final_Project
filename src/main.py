import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from graph import build_graph
from hitl import handle_escalation

load_dotenv()

# ── Welcome banner ─────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════╗
║       RAG Customer Support Assistant                 ║
║       Powered by LangGraph + Groq + ChromaDB         ║
║       Type 'quit' or 'exit' to stop                  ║
╚══════════════════════════════════════════════════════╝
"""


def run_assistant():
    print(BANNER)

    # Build the LangGraph workflow once
    app = build_graph()
    print("✅ System ready! Ask me anything about the knowledge base.\n")

    while True:
        # ── Get user input ─────────────────────────────────────────
        print("-" * 54)
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye! Have a great day.")
            break

        # ── Run through LangGraph workflow ─────────────────────────
        print("\n⏳ Thinking...\n")
        result = app.invoke({
            "question":         question,
            "context":          "",
            "answer":           "",
            "needs_escalation": False,
        })

        # ── Handle result ──────────────────────────────────────────
        if result["needs_escalation"]:
            # HITL: hand off to human agent
            final_answer = handle_escalation(
                question=question,
                bot_answer=result["answer"],
            )
            print(f"\n🧑 Agent: {final_answer}\n")
        else:
            # Direct answer from RAG
            print(f"🤖 Bot: {result['answer']}\n")


if __name__ == "__main__":
    run_assistant()