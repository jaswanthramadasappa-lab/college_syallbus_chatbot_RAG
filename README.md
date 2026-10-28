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

## 🌐 Deploy to Render

### Option A: Using Render Blueprints (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** > **Blueprint**.
3. Connect your GitHub repository (`college_syallbus_chatbot_RAG`).
4. Under Environment Variables, enter your `GEMINI_API_KEY`.
5. Click **Apply**. Render will automatically build and deploy using `render.yaml`.

### Option B: Manual Web Service
1. In Render Dashboard, click **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `python app.py`
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY` = `<your-gemini-api-key>`
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Deploy Web Service**.

---

## 🔍 How it Works

1. **Extraction & Ingestion:** Extracts text from each syllabus page using `PyMuPDF`.
2. **Chunking:** Splits document text into overlapping chunks via `RecursiveCharacterTextSplitter`.
3. **Semantic Embedding:** Embeds chunks using lightweight ONNX `all-MiniLM-L6-v2` (~120MB RAM, optimized for cloud free tiers).
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
