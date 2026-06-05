# 💬 PulseChat — AI Knowledge Base Chatbot

PulseChat is a RAG (Retrieval-Augmented Generation) chatbot that answers questions grounded in a company's internal documents. Built for Nova Tech Solutions — a fictional B2B SaaS company — it ingests company handbooks and product FAQs and lets employees ask questions in plain English.

## What is RAG?
RAG combines a vector search engine with a large language model. Instead of the LLM guessing answers from training data, it retrieves the most relevant chunks from your documents first, then uses those as context to generate an accurate, grounded response.

## Features
- Ask questions about company policies, benefits, and tools
- Ask questions about the FlowDesk product and pricing
- Source attribution — shows which document the answer came from
- Chat history within session
- Clean, minimal UI consistent with the PulseBoard design system

## Tech Stack
| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Document loading | LangChain |
| Chunking & retrieval | LangChain + ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Mistral 7B Instruct via HuggingFace Inference API |

## Run Locally

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your HuggingFace API token
cp .env.example .env
# Edit .env and add your token

# Run the app
streamlit run app.py
```

## Project Structure
```
pulsechat/
├── app.py              # Main Streamlit app and RAG pipeline
├── docs/               # Knowledge base documents
│   ├── company_handbook.txt
│   └── product_faq.txt
├── requirements.txt
├── .env.example        # Token template (never commit .env)
└── .gitignore
```

## Adding Your Own Documents
Drop any `.txt` file into the `docs/` folder and restart the app. The RAG pipeline will automatically pick it up, chunk it, and make it queryable.
