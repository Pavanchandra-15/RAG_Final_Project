import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

load_dotenv()


def get_llm():
    """Initialize Groq LLM (free tier)."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY") or "",
        temperature=0.3,
        stop_sequences=["\n\n"],
    )


# ── Prompt Template ────────────────────────────────────────────────────
ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful customer support assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I don't have enough information to answer this. Let me escalate to a human agent."

Context:
{context}

Question:
{question}

Answer:"""
)


def generate_answer(context: str, question: str) -> str:
    """Send context + question to Groq LLM and return the answer."""
    llm      = get_llm()
    prompt   = ANSWER_PROMPT.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import retrieve_relevant_chunks, format_context

    question = "What is this document about?"
    print(f"Question: {question}\n")

    chunks  = retrieve_relevant_chunks(question)
    context = format_context(chunks)
    answer  = generate_answer(context, question)

    print(f"Answer:\n{answer}")