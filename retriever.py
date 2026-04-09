import os
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_relevant_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Take a user query and return the most relevant constitutional chunks."""

    # Embed the query using the same model used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Connect to the persisted ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection("constitution")

    # Embed the query
    query_vector = embeddings.embed_query(query)

    # Search ChromaDB for top_k most similar chunks
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Format results into clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "chapter": results["metadatas"][0][i].get("chapter", ""),
            "part": results["metadatas"][0][i].get("part", ""),
            "section_number": results["metadatas"][0][i].get("section_number", ""),
            "section_title": results["metadatas"][0][i].get("section_title", ""),
            "relevance_score": round(1 - results["distances"][0][i], 4)
        })

    return chunks


if __name__ == "__main__":
    # Quick test
    test_query = "Can police search my home without permission?"
    print(f"Query: {test_query}\n")
    results = get_relevant_chunks(test_query)
    for i, chunk in enumerate(results):
        print(f"--- Result {i + 1} ---")
        print(f"Chapter: {chunk['chapter']}")
        print(f"Part: {chunk['part']}")
        print(f"Section: {chunk['section_number']} - {chunk['section_title']}")
        print(f"Relevance: {chunk['relevance_score']}")
        print(f"Text: {chunk['text'][:200]}...")
        print()