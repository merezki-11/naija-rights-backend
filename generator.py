import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for Gemini."""
    context_parts = []
    for chunk in chunks:
        citation = f"{chunk['chapter']}, {chunk['part']}, Section {chunk['section_number']}"
        context_parts.append(f"[{citation}]\n{chunk['text']}")
    return "\n\n".join(context_parts)


def generate_answer(
        query: str,
        chunks: list[dict],
        history: list[dict] = None,
        eli15: bool = False
) -> dict:
    """
    Generate a plain-language answer from retrieved chunks.

    Args:
        query: The user's question
        chunks: Retrieved constitutional chunks from retriever.py
        history: Last 5 turns of conversation [{"role": "user/assistant", "content": "..."}]
        eli15: If True, explain in very simple language for a 15-year-old

    Returns:
        dict with 'answer' and 'citations' keys
    """

    if history is None:
        history = []

    context = build_context(chunks)

    # Build the system prompt
    tone_instruction = (
        "Use very simple, friendly language that a 15-year-old Nigerian student can easily understand. "
        "Avoid all legal jargon. Use short sentences and relatable examples."
        if eli15 else
        "Use clear, plain English that any Nigerian adult can understand. "
        "Avoid excessive legal jargon but maintain accuracy."
    )

    system_prompt = f"""You are Naija Rights, an AI assistant that helps Nigerians understand their constitutional rights.

Your job is to answer questions about the Nigerian Constitution in plain, accessible language.

Rules:
1. Base your answer ONLY on the constitutional sections provided in the context below.
2. Always cite the specific Chapter and Section number in your answer.
3. If the context does not contain enough information to answer the question, say so honestly.
4. Do not make up or assume any legal information not present in the context.
5. {tone_instruction}
6. End every answer with a "📌 Source:" line listing the cited sections.

Constitutional Context:
{context}"""

    # Build the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    # Build message history
    messages = [SystemMessage(content=system_prompt)]

    for turn in history[-5:]:  # Only last 5 turns
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(HumanMessage(content=f"[Assistant]: {turn['content']}"))

    # Add current question
    messages.append(HumanMessage(content=query))

    # Call Gemini
    response = llm.invoke(messages)
    answer = response.content

    # Extract citations from chunks
    citations = []
    for chunk in chunks:
        citations.append({
            "chapter": chunk["chapter"],
            "part": chunk["part"],
            "section_number": chunk["section_number"],
            "section_title": chunk["section_title"],
            "raw_text": chunk["text"]
        })

    return {
        "answer": answer,
        "citations": citations
    }


if __name__ == "__main__":
    # Quick test
    from retriever import get_relevant_chunks

    test_query = "Can police search my home without permission?"
    print(f"Query: {test_query}\n")

    chunks = get_relevant_chunks(test_query)
    result = generate_answer(test_query, chunks)

    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== CITATIONS ===")
    for c in result["citations"]:
        print(f"- {c['chapter']}, Section {c['section_number']}: {c['section_title']}")