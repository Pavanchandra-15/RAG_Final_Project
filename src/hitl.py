import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Escalation log file ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE  = os.path.join(BASE_DIR, "escalation_log.txt")


def log_escalation(question: str, bot_answer: str, human_answer: str):
    """Save escalation details to a log file for audit purposes."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Question     : {question}\n")
        f.write(f"Bot Answer   : {bot_answer}\n")
        f.write(f"Human Answer : {human_answer}\n")
        f.write("=" * 60 + "\n\n")


def handle_escalation(question: str, bot_answer: str) -> str:
    """
    Simulate Human-in-the-Loop escalation.
    In production: send alert to Slack/email and wait for human response.
    In this project: prompt the human agent directly in the terminal.
    """
    print("\n" + "🚨 ")
    print("        HUMAN AGENT REQUIRED")
    print(f"\n📋 Customer Question  : {question}")
    print(f"🤖 Bot could not answer: {bot_answer}")
    print("\nPlease type your response as the human agent:")
    print("-" * 50)

    human_response = input("Agent Response: ").strip()

    if not human_response:
        human_response = "A human agent has been notified and will respond shortly."

    # Log the escalation
    log_escalation(question, bot_answer, human_response)
    print(f"\n✅ Human response recorded and logged to: {LOG_FILE}")

    return human_response


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing HITL module...\n")

    test_question   = "What is the CEO's personal phone number?"
    test_bot_answer = "I don't have enough information to answer this. Let me escalate to a human agent."

    final_answer = handle_escalation(test_question, test_bot_answer)
    print(f"\nFinal Answer returned to user:\n{final_answer}")