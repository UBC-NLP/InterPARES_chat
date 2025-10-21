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
git clone <repository-url>
cd InterPARES_chat
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file with your configuration:
```env
# Add your API keys and configuration here
NVIDIA_API_KEY=your_key_here
# ... other configuration
```

4. **Configure model parameters**

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

## Project Structure

```
InterPARES_chat/
├── app.py                    # Main application file
├── chat_guide.html          # User guide (HTML version)
├── README.md                # This file
├── model_params.cfg         # Model configuration
├── requirements.txt         # Python dependencies
├── modules/
│   ├── vectordb.py         # Vector database integration
│   ├── retriever.py        # Document retrieval logic
│   ├── reader.py           # LLM inference
│   └── utils.py            # Utility functions
├── json_dataset/           # Session logs (auto-created)
└── static/
    ├── ip_icon.png         # InterPARES avatar
    └── chat_icon.png       # User chat icon
```

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

## Logging

All interactions are logged in JSON format in the `json_dataset/` directory. Each log entry includes:

- Unique record and session IDs
- Timestamp and session duration
- Client location and platform information
- Query text and applied filters
- Retrieved documents
- Generated answer
- User feedback (if provided)

## Configuration

### Model Parameters (`model_params.cfg`)

Configure the retrieval and generation models:

```ini
[retriever]
MODEL=your-embedding-model

[reader]
TYPE=INF_PROVIDERS
NVIDIA_MODEL=your-llm-model
```

## API Endpoints

The application exposes several internal API endpoints:

- `/download_pdf/{filepath}`: Serves PDF files for source documents
- Session management endpoints for tracking user interactions

## Development

### Adding New Features

1. **New Filters**: Add options to `languages`, `categories`, or `phases` lists in `app.py`
2. **Custom Examples**: Modify the `QUESTIONS` dictionary to add new sample questions
3. **UI Customization**: Edit the `css` variable in `app.py` for styling changes

### Debugging

Enable detailed logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

**No results found**: Try:
- Broadening your filters (select "All")
- Rephrasing your question more specifically
- Using InterPARES-specific terminology

**Slow responses**: 
- Check your internet connection
- Verify inference provider availability
- Reduce the number of retrieved documents in configuration

*
## Acknowledgments

This project is built upon the InterPARES research initiative and its extensive documentation corpus. InterPARES explores issues related to the long-term preservation of authentic digital records.



