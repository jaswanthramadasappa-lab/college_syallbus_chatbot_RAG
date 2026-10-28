import os
import shutil
from pathlib import Path

import pymupdf as fitz
import gradio as gr
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb.utils.embedding_functions as ef


# ============================================================
# LIGHTWEIGHT ONNX EMBEDDING ADAPTER (NO PYTORCH / ~120MB RAM)
# ============================================================

class OnnxMiniLMEmbeddings(Embeddings):
    """Ultra-lightweight ONNX-based MiniLM embeddings running via ONNX Runtime without PyTorch.
    Fits easily within 512MB RAM environments like Render Free Tier (~120MB total app usage).
    """
    def __init__(self):
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return self._ef(texts)

    def embed_query(self, text):
        return self._ef([text])[0]


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "2024 Admitted Batch-B.Tech-R23-CSE.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mits_cse_syllabus"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemma-4-26b-a4b-it",
]
# Remove duplicates while preserving order
FALLBACK_MODELS = list(dict.fromkeys(FALLBACK_MODELS))

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing.\n"
        "Please provide GEMINI_API_KEY as an environment variable in Render or in a local .env file."
    )


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store():
    print("\nLoading lightweight ONNX embedding model (all-MiniLM-L6-v2)...")
    embeddings = OnnxMiniLMEmbeddings()

    # --------------------------------------------------------
    # Use existing database if valid and compatible
    # --------------------------------------------------------
    if CHROMA_DIR.exists():
        try:
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR)
            )
            count = store._collection.count()
            if count > 0:
                # Test query to verify vector dimension compatibility
                store.similarity_search("test query", k=1)
                print(f"Existing vector database verified ({count} chunks).")
                return store
        except Exception as e:
            print(f"Existing database incompatible or outdated ({e}).")
            print("Rebuilding vector database...")
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    # --------------------------------------------------------
    # Verify PDF exists
    # --------------------------------------------------------
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"\nPDF file not found:\n{PDF_PATH}\n")

    # --------------------------------------------------------
    # Read PDF using PyMuPDF
    # --------------------------------------------------------
    print("Reading syllabus PDF using PyMuPDF...")
    pdf = fitz.open(str(PDF_PATH))
    docs = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()
        if text and text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(PDF_PATH.name),
                        "page": page_number
                    }
                )
            )
    pdf.close()

    print(f"Loaded {len(docs)} pages from syllabus.")
    if not docs:
        raise ValueError("No text could be extracted from the PDF.")

    # --------------------------------------------------------
    # Split PDF into chunks
    # --------------------------------------------------------
    print("Splitting syllabus into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------------
    # Create Chroma database
    # --------------------------------------------------------
    print("\nCreating vector database...")
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # --------------------------------------------------------
    # Add documents in batches
    # --------------------------------------------------------
    batch_size = 32
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        store.add_documents(batch)
        print(f"Embedded {end}/{total} chunks")

    print("\nVector database created successfully.")
    return store


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

vector_store = build_vector_store()


# ============================================================
# LLM INVOCATION WITH FALLBACK
# ============================================================

def generate_answer(messages):
    """Invokes Gemini with automatic model fallback in case of rate limits or content filtering."""
    last_error = None

    for model_name in FALLBACK_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GEMINI_API_KEY,
                temperature=0.2
            )
            response = llm.invoke(messages)
            content = response.content

            # Parse string or list structure returned by some SDK versions
            parsed_text = ""
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", ""))
                    else:
                        text_parts.append(str(part))
                parsed_text = "".join(text_parts).strip()
            elif isinstance(content, str):
                parsed_text = content.strip()
            else:
                parsed_text = str(content).strip()

            # If finish reason is RECITATION or content is empty, try fallback model
            finish_reason = response.response_metadata.get("finish_reason") if response.response_metadata else None
            if not parsed_text and finish_reason in ["RECITATION", "SAFETY", "OTHER"]:
                print(f"Model {model_name} returned empty text with finish_reason={finish_reason}. Trying fallback...")
                continue

            if parsed_text:
                return parsed_text

        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "NOT_FOUND" in err_str:
                print(f"Model {model_name} unavailable ({err_str[:60]}...). Trying fallback...")
                continue
            else:
                raise e

    if parsed_text:
        return parsed_text
    raise RuntimeError(f"All available Gemini models failed. Last error: {last_error}")


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the official College Syllabus AI Assistant for:
Madanapalle Institute of Technology & Science (MITS)
Department of Computer Science & Engineering

You answer questions about the MITS B.Tech R23 CSE syllabus (2024 admitted batch).

GUIDELINES:
1. Use the provided syllabus context below to answer accurately, comprehensively, and informatively.
2. Synthesize and format the information clearly using your own structured words, headings, bullet points, and tables.
3. For laboratory courses or experiment lists, summarize each experiment/program clearly in an organized numbered list.
4. For course units and theory subjects, outline the topics, outcomes, and concepts covered in each unit.
5. For semester curriculum or elective lists, present them in clean, well-formatted tables or bullet lists with course codes and credits.
6. If the requested information is genuinely not available in the context, state:
   "This information is not available in the provided syllabus."

SYLLABUS CONTEXT:
{context}
"""


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(query, k=16):
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        return "", ""

    context_parts = []
    sources = []

    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            page_number = page + 1
        else:
            page_number = "Unknown"

        context_parts.append(
            f"--- [Page {page_number}] ---\n{doc.page_content}"
        )

        if page_number not in sources and page_number != "Unknown":
            sources.append(page_number)

    context = "\n\n".join(context_parts)
    source_text = ", ".join(f"Page {p}" for p in sorted(sources, key=lambda x: int(x) if str(x).isdigit() else 999))
    return context, source_text


# ============================================================
# CHAT FUNCTION
# ============================================================

def ask_syllabus(user_query, history):
    if not user_query or not user_query.strip():
        return history, ""

    query = user_query.strip()

    try:
        context, sources = retrieve_context(query, k=16)

        if not context:
            answer = "This information is not available in the provided syllabus."
        else:
            system_message = SystemMessage(
                content=SYSTEM_PROMPT.format(context=context)
            )
            user_message = HumanMessage(content=query)

            answer = generate_answer([system_message, user_message])

            if sources:
                answer += f"\n\n---\n📖 **Sources:** {sources}"

    except Exception as e:
        answer = (
            "Sorry, an error occurred while retrieving or processing your query.\n\n"
            f"**Details:** `{str(e)}`"
        )

    if history is None:
        history = []

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return history, ""


def clear_chat():
    return [], ""


# ============================================================
# GRADIO UI
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.header-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}
.header-box h1 {
    margin: 0 0 8px 0;
    font-size: 1.8rem;
    color: white;
}
.header-box p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
}
"""

with gr.Blocks(title="MITS College Syllabus Chatbot") as demo:
    gr.HTML(
        f"""
        <style>
        {CUSTOM_CSS}
        </style>
        <div class="header-box">
            <h1>🎓 MITS College Syllabus Assistant</h1>
            <p>Department of Computer Science & Engineering | B.Tech R23 (2024 Admitted Batch)</p>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        label="Chat History",
        height=500
    )

    with gr.Row():
        user_input = gr.Textbox(
            label="Ask a question about the syllabus",
            placeholder="e.g., What subjects are in II Year II Semester? Or: Explain Data Structures syllabus.",
            lines=1,
            scale=5,
            autofocus=True
        )
        send_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    gr.Markdown("### 💡 Sample Questions to Click & Try:")
    with gr.Row():
        q1 = gr.Button("📘 II Year II Sem Subjects", size="sm")
        q2 = gr.Button("📗 DBMS Course Details", size="sm")
        q3 = gr.Button("📙 Professional Electives", size="sm")
        q4 = gr.Button("📕 Skill Enhancement Courses", size="sm")

    # Wire buttons
    send_btn.click(
        ask_syllabus,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )

    user_input.submit(
        ask_syllabus,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )

    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, user_input]
    )

    q1.click(lambda h: ask_syllabus("What subjects are in II Year II Semester?", h), inputs=[chatbot], outputs=[chatbot, user_input])
    q2.click(lambda h: ask_syllabus("What is the syllabus and credits for Database Management Systems?", h), inputs=[chatbot], outputs=[chatbot, user_input])
    q3.click(lambda h: ask_syllabus("List the Professional Elective courses available in the syllabus.", h), inputs=[chatbot], outputs=[chatbot, user_input])
    q4.click(lambda h: ask_syllabus("What are the Skill Enhancement Courses in the curriculum?", h), inputs=[chatbot], outputs=[chatbot, user_input])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n==============================================")
    print("   MITS COLLEGE SYLLABUS CHATBOT")
    print("==============================================\n")
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)