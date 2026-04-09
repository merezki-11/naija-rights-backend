import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import time
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH = os.path.join(os.path.dirname(__file__), "constitution", "constitution.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def extract_chunks_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF and organize into chunks by section."""
    doc = fitz.open(pdf_path)
    chunks = []
    current_chapter = "Unknown"
    current_part = "Unknown"
    current_section_number = ""
    current_section_title = ""
    current_text_lines = []

    def save_chunk():
        text = " ".join(current_text_lines).strip()
        if text and len(text) > 50:
            chunks.append({
                "chapter": current_chapter,
                "part": current_part,
                "section_number": current_section_number,
                "section_title": current_section_title,
                "text": text
            })

    for page in doc:
        lines = page.get_text().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("CHAPTER") and len(line) < 60:
                save_chunk()
                current_chapter = line
                current_text_lines = []

            elif line.startswith("PART") and len(line) < 60:
                save_chunk()
                current_part = line
                current_text_lines = []

            elif line and line[0].isdigit() and "." in line[:5]:
                save_chunk()
                parts = line.split(".", 1)
                current_section_number = parts[0].strip()
                current_section_title = parts[1].strip() if len(parts) > 1 else ""
                current_text_lines = [line]

            else:
                current_text_lines.append(line)

    save_chunk()
    doc.close()
    print(f"Extracted {len(chunks)} chunks from PDF")
    return chunks


def embed_and_store(chunks: list[dict]):
    """Embed chunks using Gemini and store in ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    try:
        client.delete_collection("constitution")
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection("constitution")

    print(f"Embedding and storing {len(chunks)} chunks...")

    batch_size = 30
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [{
            "chapter": c["chapter"],
            "part": c["part"],
            "section_number": c["section_number"],
            "section_title": c["section_title"]
        } for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        vectors = embeddings.embed_documents(texts)

        collection.add(
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        current_batch = i // batch_size + 1
        print(f"Stored batch {current_batch} / {total_batches}")

        if current_batch < total_batches:
            print(f"Waiting 65 seconds before next batch...")
            time.sleep(65)

    print(f"Done. {collection.count()} chunks stored in ChromaDB.")


if __name__ == "__main__":
    print("Starting ingestion...")
    chunks = extract_chunks_from_pdf(PDF_PATH)
    embed_and_store(chunks)
    print("Ingestion complete.")