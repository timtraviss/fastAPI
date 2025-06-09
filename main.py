from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import openai
from pinecone import Pinecone
from uuid import uuid4
from datetime import datetime
import hashlib
import re
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import io

load_dotenv()

# Add this after loading environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print(f"Pinecone API key loaded: {pinecone_api_key[:8]}..." if pinecone_api_key else "No Pinecone API key found")

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class ProcessResponse(BaseModel):
    success: bool
    message: str
    file_hash: Optional[str] = None
    pages: Optional[int] = None
    chunks: Optional[int] = None

class SearchResponse(BaseModel):
    answer: str
    sources: List[Dict]
    context_found: bool

class FileInfo(BaseModel):
    filename: str
    file_hash: str
    pages: int
    chunks: int
    upload_date: str

# Initialize FastAPI
app = FastAPI(
    title="PDF Q&A API",
    description="Upload PDFs and ask questions using AI-powered search",
    version="1.0.0"
)

# Initialize APIs
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found")
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found")

    # Initialize Pinecone with newer API
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("n8npdffiles")
except Exception as e:
    print(f"Failed to initialize APIs: {str(e)}")
    raise

# In-memory storage for file tracking (use Redis in production)
uploaded_files = {}

def get_file_hash(file_bytes: bytes) -> str:
    """Generate a unique hash for the file to prevent duplicates."""
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """Extract text from PDF and return text with page count."""
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages_text.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(pages_text), len(doc)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")
    finally:
        if doc:
            doc.close()

def chunk_text_smart(text: str, max_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Smart chunking that preserves sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        if current_words + sentence_words > max_size and current_chunk:
            page_match = re.search(r'\[Page (\d+)\]', current_chunk)
            page_num = int(page_match.group(1)) if page_match else None
            
            chunks.append({
                "text": current_chunk.strip(),
                "word_count": current_words,
                "page": page_num
            })
            
            overlap_words = current_chunk.split()[-overlap:] if overlap > 0 else []
            current_chunk = " ".join(overlap_words) + " " + sentence
            current_words = len(overlap_words) + sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_words += sentence_words
    
    if current_chunk.strip():
        page_match = re.search(r'\[Page (\d+)\]', current_chunk)
        page_num = int(page_match.group(1)) if page_match else None
        
        chunks.append({
            "text": current_chunk.strip(),
            "word_count": current_words,
            "page": page_num
        })
    
    return chunks

async def embed_and_store(chunks: List[Dict], filename: str, file_hash: str) -> bool:
    """Embed chunks and store in Pinecone."""
    try:
        for i, chunk_data in enumerate(chunks):
            # Create embedding
            embedding_response = openai.embeddings.create(
                input=[chunk_data["text"]], 
                model="text-embedding-3-small"
            )
            embedding = embedding_response.data[0].embedding
            
            # Prepare metadata
            metadata = {
                "chunk": chunk_data["text"],
                "filename": filename,
                "file_hash": file_hash,
                "page": chunk_data.get("page"),
                "word_count": chunk_data["word_count"],
                "upload_date": datetime.now().isoformat()
            }
            
            # Upsert to Pinecone
            index.upsert([{
                "id": f"{file_hash}_{i}",
                "values": embedding,
                "metadata": metadata
            }])
        
        return True
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding and storage: {str(e)}")

def search_documents(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """Search for relevant chunks and return context with source info."""
    try:
        # Create question embedding
        embedding_response = openai.embeddings.create(
            input=[question], 
            model="text-embedding-3-small"
        )
        question_embedding = embedding_response.data[0].embedding
        
        # Query Pinecone
        results = index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Process results
        context_parts = []
        sources = []
        
        for match in results["matches"]:
            if "metadata" in match and "chunk" in match["metadata"]:
                metadata = match["metadata"]
                similarity_score = match["score"]
                
                context_parts.append(metadata["chunk"])
                sources.append({
                    "filename": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page"),
                    "score": similarity_score,
                    "chunk_preview": metadata["chunk"][:100] + "..." if len(metadata["chunk"]) > 100 else metadata["chunk"]
                })
        
        return "\n\n".join(context_parts), sources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

def generate_answer(context: str, question: str) -> str:
    """Generate answer using OpenAI."""
    try:
        prompt = f"""Answer the question based on the provided context. If the context doesn't contain enough information to fully answer the question, say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Be specific and cite relevant details from the context
- If information is missing, acknowledge what cannot be answered
- Keep the answer concise but comprehensive
- Use plain text formatting only"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        answer = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\$\%\n]', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "PDF Q&A API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload", response_model=ProcessResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read file
    file_bytes = await file.read()
    
    # Check file size (10MB limit)
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
    
    # Generate file hash
    file_hash = get_file_hash(file_bytes)
    
    # Check for duplicates
    if file_hash in [info['file_hash'] for info in uploaded_files.values()]:
        return ProcessResponse(
            success=False,
            message=f"File '{file.filename}' already exists",
            file_hash=file_hash
        )
    
    # Extract text
    text, page_count = extract_text_from_pdf(file_bytes)
    
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Chunk text
    chunks = chunk_text_smart(text)
    
    # Embed and store
    await embed_and_store(chunks, file.filename, file_hash)
    
    # Store file info
    uploaded_files[file.filename] = {
        'file_hash': file_hash,
        'pages': page_count,
        'chunks': len(chunks),
        'upload_date': datetime.now().isoformat()
    }
    
    return ProcessResponse(
        success=True,
        message=f"Successfully processed '{file.filename}'",
        file_hash=file_hash,
        pages=page_count,
        chunks=len(chunks)
    )

@app.post("/question", response_model=SearchResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents."""
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    
    # Search documents
    context, sources = search_documents(request.question, request.top_k)
    
    if not context:
        return SearchResponse(
            answer="No relevant information found in the uploaded documents.",
            sources=[],
            context_found=False
        )
    
    # Generate answer
    answer = generate_answer(context, request.question)
    
    return SearchResponse(
        answer=answer,
        sources=sources,
        context_found=True
    )

@app.get("/files", response_model=List[FileInfo])
async def list_files():
    """List all uploaded files."""
    return [
        FileInfo(
            filename=filename,
            file_hash=info['file_hash'],
            pages=info['pages'],
            chunks=info['chunks'],
            upload_date=info['upload_date']
        )
        for filename, info in uploaded_files.items()
    ]

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific file."""
    if filename not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[filename]
    file_hash = file_info['file_hash']
    
    # Delete from Pinecone (you'd need to implement this based on your needs)
    # For now, just remove from local storage
    del uploaded_files[filename]
    
    return {"message": f"File '{filename}' deleted successfully"}

@app.delete("/files")
async def clear_all_files():
    """Clear all uploaded files."""
    uploaded_files.clear()
    return {"message": "All files cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)