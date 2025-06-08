# PDF Q&A API

A FastAPI application that allows users to upload PDF documents and ask questions about their content using AI-powered search with OpenAI embeddings and Pinecone vector database.

## Features

- Upload PDF documents (up to 10MB)
- Smart text chunking with sentence boundary preservation
- Vector embeddings using OpenAI's text-embedding-3-small
- Semantic search with Pinecone
- AI-powered question answering using GPT-4
- Duplicate file detection
- RESTful API with automatic documentation

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /upload` - Upload PDF file
- `POST /question` - Ask questions about uploaded documents
- `GET /files` - List all uploaded files
- `DELETE /files/{filename}` - Delete specific file
- `DELETE /files` - Clear all files

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
4. Run the application:
   ```bash
   uvicorn fastapi_pdf_qa:app --reload
   ```

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `PINECONE_API_KEY` - Your Pinecone API key

## Deployment to Render

This application is configured for deployment to Render using Docker.

1. Push your code to GitHub
2. Connect your repository to Render
3. Set the environment variables in Render dashboard
4. Deploy using the provided `render.yaml` configuration

## Requirements

- Python 3.11+
- OpenAI API access
- Pinecone vector database
- PDF files for processing

## API Documentation

Once running, visit `/docs` for interactive API documentation or `/redoc` for alternative documentation format.