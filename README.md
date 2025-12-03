# InterPARES-Chat 💬

An AI-powered conversational assistant designed to help users explore and understand InterPARES (International Research on Permanent Authentic Records in Electronic Systems) documents through natural language queries.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-demos.dlnlp.ai-blue)](http://demos.dlnlp.ai/InterPARES/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## 🌐 Live Demo

Try InterPARES-Chat online: **[demos.dlnlp.ai/InterPARES/](http://demos.dlnlp.ai/InterPARES/)**

No installation required - access the full functionality through your web browser!

---



## Overview

InterPARES-Chat enables users to interact with the comprehensive corpus of InterPARES documentation—including reports, guidelines, case studies, and theoretical frameworks—using natural language questions. The system retrieves relevant information from the document collection and generates clear, structured answers with proper citations.

## Features

- **Natural Language Queries**: Ask questions in plain English about InterPARES concepts, methodologies, and findings
- **Source Citation**: All answers include references to specific documents with downloadable PDFs
- **Advanced Filtering**: Filter documents by language, category, phase, and specific reports
- **Session Management**: Tracks user sessions with duration, location, and platform information
- **Feedback System**: Users can provide feedback on answer quality
- **Streaming Responses**: Real-time AI-generated responses with source attribution
- **Modern UI**: Clean, responsive interface with light theme optimized for readability

## Technology Stack

- **Frontend**: Gradio with custom CSS and Tailwind styling
- **Vector Database**: Qdrant for semantic search
- **LLM Integration**: Supports inference provider endpoints (vLLM, NVIDIA)
- **Backend**: FastAPI (via Gradio) for serving PDFs and handling requests
- **Logging**: Comprehensive session and interaction logging in JSON format

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/UBC-NLP/InterPARES_chat
cd InterPARES_chat
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```



3. **Configure model parameters**

Edit `model_params.cfg` to set your preferred:
- Retriever model
- Reader model and type
- Inference provider settings

## Usage

### Starting the Application

```bash
python app.py
```

The application will launch and provide a local URL (typically `http://127.0.0.1:7860`) and optionally a public share URL.

### Asking Questions

For best results, follow these guidelines:

#### ✅ Effective Questions

- **Be specific**: "What are the benchmark requirements supporting the presumption of authenticity of electronic records in InterPARES 1?"
- **Include context**: "How does InterPARES define the concepts of identity and integrity in assessing authenticity?"
- **Reference phases**: "What definitions of digital preservation emerged from the InterPARES survey?"

#### ❌ Less Effective Questions

- Too vague: "What is authenticity?"
- Too broad: "Tell me about preservation"
- Lacking context: "How do we trust records?"

### Using Filters

Filter your document search by:

- **Language**: English, Italian, German, French, Spanish, and more
- **Category**: Dissemination, archival, book, policy, report, research, etc.
- **Phase**: InterPARES Phase 1, 2, 3, or 4
- Select "All" to include all options for a filter

### Example Questions

The application includes curated sample questions covering:

- Authenticity assessment requirements
- Identity and integrity concepts
- Preservation procedures
- Appraisal methodologies
- Selection processes
- Digital preservation strategies

Click any example to automatically populate the input field.


## Key Components

### Vector Database (`modules/vectordb.py`)
- Manages Qdrant collections for semantic search
- Handles document embeddings and metadata

### Retriever (`modules/retriever.py`)
- Implements context retrieval with metadata filtering
- Supports multi-filter queries (language, category, phase)

### Reader (`modules/reader.py`)
- Interfaces with LLM inference providers
- Handles streaming responses

### Session Management
- Tracks user sessions with unique IDs
- Records session duration, client location, and platform info
- Logs all interactions for analysis

### Feedback System
- Collects user feedback (👍/👎) on responses
- Associates feedback with specific queries and answers
- Stores feedback in session logs

## Used Model Cards

### Models Functionality in the Pipeline

| Model | Role | Description |
|-------|------|-------------|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Retriever | Generates dense embeddings for semantic search, enabling retrieval of relevant document chunks from the vector database |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Retriever (Alternative) | Lightweight embedding model for semantic search with lower resource requirements |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reranker | Re-scores retrieved documents to improve ranking precision before presenting to the reader |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Reranker (Alternative) | Lightweight reranker model for document re-scoring with lower resource requirements |
| [Qwen3-4B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-4B) | Reader | Generates natural language answers based on retrieved context, with streaming response capability |

### BAAI/bge-m3

**Model Card**: [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

#### Limitations and Considerations
- Maximum sequence length of 8192 tokens; longer documents require chunking
- Multilingual but may have varying performance across languages
- Embedding quality depends on domain similarity to training data
- Resource-intensive for large-scale indexing operations

### BAAI/bge-reranker-v2-m3

**Model Card**: [https://huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

#### Limitations and Considerations
- Cross-encoder architecture increases latency compared to bi-encoders
- Best suited for reranking a limited number of candidates (top-k from retriever)
- May not generalize well to highly specialized or out-of-domain terminology
- Requires GPU for efficient inference at scale

### Qwen/Qwen3-Embedding-0.6B

**Model Card**: [https://huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

#### Limitations and Considerations
- Smaller model size (0.6B) may result in lower embedding quality compared to larger models
- May have reduced performance on complex or domain-specific queries
- Suitable for resource-constrained environments where bge-m3 is too large
- Multilingual support may vary in quality across languages

### Qwen/Qwen3-Reranker-0.6B

**Model Card**: [https://huggingface.co/Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

#### Limitations and Considerations
- Smaller model size trades accuracy for speed and lower memory usage
- May have reduced precision on nuanced relevance distinctions
- Suitable for resource-constrained environments where bge-reranker-v2-m3 is too large
- Performance gap with larger rerankers may be noticeable on specialized domains

### Qwen3-4B-Instruct-2507-FP8

**Model Card**: [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

#### Limitations and Considerations
- FP8 quantization trades some accuracy for improved inference speed and reduced memory
- Context window limitations may affect handling of very long retrieved passages
- May hallucinate information not present in the provided context
- Response quality depends on prompt engineering and context relevance
- Requires appropriate hardware support for FP8 inference


## API Endpoints

The application exposes several internal API endpoints:

- `/download_pdf/{filepath}`: Serves PDF files for source documents
- Session management endpoints for tracking user interactions


## Acknowledgments

This project is built upon the InterPARES research initiative and its extensive documentation corpus. InterPARES explores issues related to the long-term preservation of authentic digital records.



