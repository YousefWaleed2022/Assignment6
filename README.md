
# 🔍 RAG System: Retrieval-Augmented Generation with Local Vector Indexing

This project presents a fully functional RAG (Retrieval-Augmented Generation) system built using open-source tools, with a focus on local vector-based storage. The goal is to enhance large language model (LLM) responses by pulling in relevant external knowledge during generation.

---

## 📌 Overview

This system offers:

- Support for parsing and chunking documents in formats like PDF, DOCX, and TXT  
- Local vector indexing powered by **FAISS** for fast similarity-based retrieval  
- Multiple retrieval modes including standard similarity, MMR (Maximum Marginal Relevance), and hybrid  
- Seamless integration with LLM APIs for response generation  
- An evaluation framework to assess both retrieval and generation accuracy  
- A simple interactive interface for querying the system  

---

## 🧱 System Components

### 📄 Document Handling

- **DocumentLoader**: Reads files from supported formats  
- **TextChunker**: Breaks documents into manageable segments  

### 🔗 Embeddings & Vector Storage

- **Embedder**: Generates semantic embeddings via Sentence Transformers  
- **FAISS Index**: Stores and retrieves vectors for efficient matching  

### 🔍 Retrieval

- **VectorRetriever**: Supports several strategies:
  - Pure similarity search
  - MMR for diversity
  - Hybrid strategy for combined strengths  

### 🧠 Language Model Integration

- **GroqProvider**: Connects with Groq’s LLM API  
- Includes configurable prompts for tailored outputs  

### 📊 Evaluation Tools

- **RAGEvaluator**: Measures retrieval quality (precision, recall) and generation quality (keyword hits, accuracy)  
- Designed to benchmark different setups  

---

## ⚙️ Getting Started

### ✅ Prerequisites

- Python 3.10 or higher  
- API key for a language model provider (Groq is pre-configured)  

### 📥 Installation

Clone the repo:

```bash
git clone https://github.com/YousefWaleed2022/Assignment6
cd Assignment6
```

Set up the environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Configure API access:

```bash
# Inside .env file
GROQ_API_KEY=your_api_key_here
```

---

## 🚀 How to Run

Basic launch:

```bash
python rag_system.py
```

This will check for previously indexed documents or ingest new ones from the `documents` directory.

Additional options:

- Force document ingestion:  
  `python rag_system.py --ingest`  

- Run evaluation mode:  
  `python rag_system.py --evaluate`  

- Rebuild vector index from scratch:  
  `python rag_system.py --clean`  

- Direct query mode:  
  `python rag_system.py --query "What is retrieval-augmented generation?"`  

- Use a custom folder:  
  `python rag_system.py --data-dir path/to/docs`  

---

## 📚 Document Support

You can use:

- `.pdf` files  
- `.docx` Word documents  
- `.txt` files  

Place them inside the `documents` folder or specify another directory using `--data-dir`.

---

## 📈 Experimental Highlights

### 🔁 Retrieval Strategy Comparison

| Method       | Pros                        | Cons                                 |
|--------------|-----------------------------|--------------------------------------|
| Similarity   | Fast, simple                | May return redundant results         |
| MMR          | More diverse answers        | Slightly slower                      |
| Hybrid       | Balanced, best overall      | Higher compute cost                  |

**Hybrid** performs best when queries are complex or require broader context.

---

## 🔢 Embedding Models

Tested Model: `all-MiniLM-L6-v2` from Sentence Transformers  

- ✅ Lightweight and fast  
- ⚠️ Slightly reduced semantic precision  

---

## 📏 Evaluation Criteria

- **Retrieval**: Precision, recall, F1  
- **Generation**: Keyword match, accuracy  
- **Edge Cases**: Handles unusual or invalid queries gracefully  

---

## ✅ Pros & Cons

### ✔️ Strengths:

- Local FAISS-based indexing = fast retrieval  
- Versatile retrieval strategies  
- Built-in evaluation framework  
- Easy-to-use interactive mode  
- Clear error handling and logging  

### ❌ Limitations:

- Only supports text documents  
- No dynamic KB updates (yet)  
- Single LLM provider (Groq)  
- No query rewriting or self-correction currently  

---

## 🧪 Challenges & Fixes

| Challenge                              | Solution                                                   |
|----------------------------------------|-------------------------------------------------------------|
| Large-scale corpora + memory pressure | Batched embedding + parallel loading                        |
| Relevance vs. diversity                | Hybrid method: combines similarity + MMR                   |
| Preserving chunk context               | Added chunk overlap + metadata retention                   |

---

## 🔮 Future Enhancements

- Smarter retrieval via query rewriting  
- Live updates to the knowledge base  
- Built-in self-evaluation with feedback  
- Plug-in support for more LLM providers  
- Expand doc support (Markdown, HTML, etc.)  

---

## 🙌 Credits

- Built for **CSAI 422 - Lab Assignment 6**  
- Uses **LangChain**, **Sentence Transformers**, and **FAISS**  
- Inspired by leading research in RAG techniques  
