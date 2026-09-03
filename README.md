# MultiAgentRAG

MultiAgentRAG is a multi-agent retrieval-augmented generation system for answering legal questions about divorce and inheritance law in Italy, Slovenia, and Estonia. A supervisor routes each question to one or more specialist agents, which search a shared Pinecone index, rerank the retrieved material, and produce source-grounded answers through a local Gradio interface.

> **Important:** This project is an experimental legal-information tool. Its output is not legal advice and should be independently verified before it is used to make legal decisions.

## Features

- LLM-based triage between direct responses and document retrieval
- Eleven specialist agents covering legislation and case law across three jurisdictions
- Metadata-filtered semantic retrieval from a shared Pinecone index
- BGE-M3 embeddings and cross-encoder reranking
- Parallel execution when several specialist agents are selected
- Per-citation grounding checks with one corrective pass when needed
- Ten-turn, in-memory conversational context
- Cross-session semantic memory backed by ChromaDB
- Full chat-session history backed by SQLite
- Local Gradio interface with previous-chat navigation
- Configurable OpenAI-compatible LLM providers: Groq, DeepSeek, Gemini, and z.ai

## How it works

```text
User question
    |
    v
Supervisor: triage and routing
    |------------------------------|
    | direct answer                | retrieval required
    v                              v
LLM response              Selected specialist agents
                                  |
                                  v
                       Pinecone retrieval (top 20)
                                  |
                                  v
                       Cross-encoder reranking (top 5)
                                  |
                                  v
                         Partial grounded answers
                                  |
                                  v
                      Aggregation and citation check
                                  |
                                  v
                              Final answer
```

All specialist agents query the same `legal-rag` Pinecone index. The registry in `config.py` restricts each query by jurisdiction, legal area, and document type. Estonia's inheritance agent is the only combined agent: it covers both legislation and case law because that case-law collection is comparatively small.

## Project structure

| Path | Purpose |
| --- | --- |
| `ui.py` | Local Gradio chat interface and application entry point |
| `agents.py` | Supervisor, specialist-agent retrieval, reranking, aggregation, and memory coordination |
| `config.py` | Specialist-agent registry and Pinecone metadata filters |
| `llm_client.py` | LLM provider selection and provider-specific model configuration |
| `guardrails/output_guard.py` | Per-citation support verification and answer correction |
| `memory/short_term.py` | Ten-turn in-memory sliding window |
| `memory/long_term.py` | Persistent semantic memory in ChromaDB |
| `memory/chat_history.py` | Full session transcripts in SQLite |
| `Ingestion.ipynb` | Dataset validation, chunking, embedding, and Pinecone ingestion workflow |
| `Contest_Data.zip` | Packaged source dataset |

The application creates `chroma_db/` for long-term memory and `chat_history.db` for complete chat transcripts at runtime.

## Prerequisites

- Python 3.10 or newer
- A Pinecone account and a `legal-rag` index configured with:
  - 1,024 dimensions
  - cosine similarity
- An API key for at least one supported LLM provider
- Sufficient memory to load BGE-M3 and `bge-reranker-v2-m3` locally
- A Hugging Face token if the model download requires authentication
- A CUDA-capable GPU is strongly recommended for running the ingestion notebook

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the runtime dependencies:

```bash
python -m pip install chromadb gradio openai pinecone python-dotenv sentence-transformers certifi
```

To run the ingestion notebook, also install Jupyter, PyTorch, and tqdm:

```bash
python -m pip install jupyter torch tqdm
```

No pinned dependency file is currently included, so use an isolated environment to avoid conflicts with other projects.

## Configuration

Create `Apikey.env` in the repository root. Select one provider with `LLM_PROVIDER` and provide its corresponding key:

```dotenv
LLM_PROVIDER=groq

GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
ZAI_API_KEY=your_zai_key

PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_hugging_face_token

# Optional: enable verbose application logging.
LEGAL_RAG_DEBUG=false
```

Valid values for `LLM_PROVIDER` are `groq`, `deepseek`, `gemini`, and `zai`. Only the key for the selected provider is required, together with `PINECONE_API_KEY`. Do not commit `Apikey.env`; it is excluded by `.gitignore`.

Provider model assignments are defined centrally in `llm_client.py`. The retrieval embedding model in `agents.py` must remain identical to the model used during ingestion.

## Build the Pinecone index

Skip this section if the `legal-rag` index has already been populated with the supplied dataset and metadata schema.

1. Create the Pinecone index described in [Prerequisites](#prerequisites).
2. Extract `Contest_Data.zip`, or allow the notebook to extract it when `Contest_Data/` is absent.
3. Open `Ingestion.ipynb` in a GPU-backed Jupyter environment.
4. Run the validation and preview cells, then review all reported errors, unknown law values, and document counts.
5. Run the final ingestion cell only after confirming the preview.

> **Warning:** `RESET_NAMESPACE_BEFORE_UPSERT` is set to `True` in the notebook. The final ingestion step therefore clears the configured Pinecone namespace before uploading vectors. Set it to `False` if existing vectors must be preserved.

The ingestion workflow uses 1,500-token chunks with a 150-token overlap. It stores normalized BGE-M3 embeddings, source metadata, raw chunk text, stable vector IDs, and citation labels in Pinecone's default namespace.

## Run the application

From the repository root, with the virtual environment active:

```bash
python ui.py
```

The first launch may take time while the embedding and reranking models are downloaded and loaded. Open the local address printed by Gradio, enter a legal question, and use the sidebar to start a new chat or reopen an earlier session.

The interface is intentionally local-only (`share=False`). A public deployment would require per-user state isolation, authentication, secret management, and production hardening.

## Agent coverage

| Jurisdiction | Divorce | Inheritance |
| --- | --- | --- |
| Italy | Separate case-law and legislation agents | Separate case-law and legislation agents |
| Slovenia | Separate case-law and legislation agents | Separate case-law and legislation agents |
| Estonia | Separate case-law and legislation agents | One combined case-law and legislation agent |

To change routing coverage or add an agent, update `AGENT_REGISTRY` in `config.py` and ensure that its `pinecone_filter` uses the exact metadata values written by `Ingestion.ipynb`.

## Memory and persistence

- **Short-term memory:** keeps the latest ten completed turns in RAM for the active session.
- **Long-term memory:** stores concise summaries of retrieval-grounded answers in `chroma_db/` and recalls similar past questions for the relevant agents. It supplements, but never replaces, current document retrieval.
- **Chat history:** stores complete, unmodified session transcripts in `chat_history.db` for the previous-chats sidebar.

Direct, non-retrieval answers are saved to chat history and short-term memory, but not to long-term semantic memory.

## Troubleshooting

- **`LLM_PROVIDER Missing in Apikey.env`:** add a valid `LLM_PROVIDER` entry to `Apikey.env`.
- **Missing provider API key:** add the key associated with the selected provider.
- **Missing `PINECONE_API_KEY`:** add the Pinecone key to `Apikey.env`.
- **Pinecone dimension error:** recreate or reconfigure `legal-rag` as a 1,024-dimensional cosine index.
- **Poor or empty retrieval:** confirm that the index is populated and that its `country`, `law`, and `doc_type` metadata values match `config.py`.
- **Slow first startup:** model downloads and initialization are expected on the first run; a GPU can substantially reduce model inference time.
- **Certificate error during Gradio import:** install or update `certifi`. The UI already repairs a stale `SSL_CERT_FILE` when a valid certificate bundle is available.

## Security and operational notes

- Keep all API credentials in `Apikey.env` and out of version control.
- The citation guardrail is a secondary safety mechanism, not a guarantee of legal accuracy.
- Inspect and back up `chat_history.db` and `chroma_db/` according to the sensitivity of user conversations.
- Review Pinecone usage and LLM-provider costs before exposing the application to sustained traffic.
