# 🎓 MITS College Syllabus Chatbot

An AI-powered RAG (Retrieval-Augmented Generation) chatbot that accurately answers questions from the official **Madanapalle Institute of Technology & Science (MITS)** B.Tech R23 Computer Science & Engineering syllabus PDF (2024 Admitted Batch).

---

## 🚀 Tech Stack

- **Framework & UI:** Gradio 6.x
- **LLM:** Google Gemini (`gemini-3.5-flash-lite` / `gemini-flash-lite-latest` with automatic multi-model fallback)
- **Embeddings:** HuggingFace `sentence-transformers/all-mpnet-base-v2` (768-dimensional high-precision semantic embeddings)
- **Vector Database:** Chroma DB (persistent vector store)
- **PDF Extraction:** PyMuPDF (`pymupdf`)
- **RAG Orchestration:** LangChain (`langchain`, `langchain-google-genai`, `langchain-chroma`, `langchain-huggingface`)

---

## 🛠️ Setup & Installation

### 1. Virtual Environment

Using PowerShell on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy `.env.example` to `.env` and insert your Gemini API Key:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 4. Run the Application

```bash
python app.py
```

The Gradio web interface will launch locally (e.g., `http://127.0.0.1:7860`).

---

## 🔍 How it Works

1. **Extraction & Ingestion:** Extracts text from each syllabus page using `PyMuPDF`.
2. **Chunking:** Splits document text into overlapping chunks via `RecursiveCharacterTextSplitter`.
3. **Semantic Embedding:** Embeds chunks using `sentence-transformers/all-mpnet-base-v2`.
4. **Vector Store:** Saves and indexes chunks in Chroma vector storage with auto-healing dimension verification.
5. **Context Retrieval:** Retrieves top relevant syllabus passages with source page numbers.
6. **Grounding & Response:** Passes retrieved syllabus chunks into Gemini to provide grounded, hallucination-free answers citing page numbers.

---

## 💡 Example Questions

- *What subjects are in II Year II Semester?*
- *What is the syllabus and credits for Database Management Systems?*
- *List the Professional Elective courses.*
- *What are the Skill Enhancement Courses in the curriculum?*
- *What are the course outcomes for Introduction to Programming?*
- *What are the units in Advanced Data Structures and Algorithm Analysis?*
