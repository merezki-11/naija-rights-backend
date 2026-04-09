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

    tone_instruction = (
        "Explain it extremely simply like you're talking to a 15-year-old Nigerian student. "
        if eli15 else
        "Talk like a knowledgeable, street-smart Nigerian legal advisor speaking to a guy on the street. "
        "You MUST include at least one or two natural Nigerian Pidgin phrases (e.g., 'See ehn', 'Make we no lie', 'Abeg', 'Omo', 'Wahala') so you don't sound like a boring robot."
    )

    system_prompt = f"""You are Naija Rights, a friendly, street-smart AI legal companion for everyday Nigerians.

Your job is to break down the Nigerian Constitution so that anyone on the street can easily understand it.

Rules:
1. Base your answer ONLY on the constitutional context below. If it's not there, honestly say "I no fit find the exact law for this one right now."
2. Keep your answer EXTREMELY short and punchy. Maximum 2-3 sentences per point. 
3. Use everyday Naija examples (e.g. Police checkpoints, street matters) if it helps explain the law.
4. {tone_instruction} Write very conversationally, never like a stuffy textbook.
5. Always end your response EXACTLY with this format: "📌 Source: The Nigerian Constitution, Chapter [X], Part [Y], Section [Z]".

Constitutional Context:
{context}"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    messages = [SystemMessage(content=system_prompt)]

    for turn in history[-5:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=str(turn["content"])))
        else:
            messages.append(HumanMessage(content=f"[Assistant]: {str(turn['content'])}"))

    messages.append(HumanMessage(content=str(query)))

    response = llm.invoke(messages)
    answer = response.content

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