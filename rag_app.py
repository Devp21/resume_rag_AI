from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
import time
import tempfile
import hashlib
import uuid
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 3: Configuration (Updated for your index)
# ============================================================================
class Config:
    # Pinecone settings (from your console)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "resume-rag"  # Your existing index
    
    # Groq settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Embedding settings - Match your index exactly
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
    EMBEDDING_DIMENSION = 768  # Your index dimension
    
    # Processing settings
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 4
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Validate required environment variables
    @classmethod
    def validate(cls):
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment")
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")

# ============================================================================
# STEP 4: Pydantic Models
# ============================================================================
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    relevance_scores: List[float]

class UploadResponse(BaseModel):
    message: str
    user_id: str
    chunks_created: int
    file_name: str

class HealthResponse(BaseModel):
    status: str
    pinecone_connected: bool
    groq_connected: bool

# ============================================================================
# STEP 5: RAG System Class
# ============================================================================
class ResumeRAGSystem:
    def __init__(self):
        Config.validate()
        self.config = Config()
        self.pc = None
        self.index = None
        self.embeddings = None
        self.llm = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all components"""
        try:
            # 1. Initialize Pinecone
            self.pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
            self.index = self.pc.Index(self.config.INDEX_NAME)
            logger.info("✅ Pinecone connected")
            
            # 2. Setup embeddings - FREE sentence-transformers
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings model loaded")
            
            # 3. Setup LLM - FREE Groq
            self.llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.1,
                groq_api_key=self.config.GROQ_API_KEY
            )
            logger.info("✅ Groq LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {str(e)}")
            raise
    
    def process_pdf(self, file_path: str, user_id: str) -> int:
        """Process PDF and store in Pinecone"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in PDF")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Prepare vectors
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue
                
                # Create embedding
                embedding = self.embeddings.embed_query(chunk.page_content)
                
                # Create unique ID
                chunk_id = f"{user_id}_{hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]}_{i}"
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.page_content,
                        "user_id": user_id,
                        "page": chunk.metadata.get("page", 0),
                        "source": os.path.basename(file_path)
                    }
                }
                vectors_to_upsert.append(vector)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"✅ Processed {len(vectors_to_upsert)} chunks for user {user_id}")
            return len(vectors_to_upsert)
            
        except Exception as e:
            logger.error(f"❌ Failed to process PDF: {str(e)}")
            raise
    
    def ask_question(self, question: str, user_id: str) -> dict:
        """Ask question and return detailed response"""
        try:
            # Create embedding
            question_embedding = self.embeddings.embed_query(question)
            
            # Search Pinecone
            search_results = self.index.query(
                vector=question_embedding,
                top_k=self.config.TOP_K_RESULTS,
                filter={"user_id": user_id},
                include_metadata=True
            )
            
            if not search_results.matches:
                return {
                    "answer": "❌ No relevant information found in your resume. Please upload your resume first.",
                    "sources": [],
                    "relevance_scores": []
                }
            
            # Extract context and metadata
            context_chunks = []
            sources = []
            scores = []
            
            for match in search_results.matches:
                context_chunks.append(match.metadata['text'])
                sources.append(f"Page {match.metadata.get('page', 'N/A')}")
                scores.append(round(float(match.score), 3))
            
            context = "\n\n".join(context_chunks)
            
            # Generate response
            prompt = f"""Based on the following resume content, provide a comprehensive and accurate answer.

Resume Content:
{context}

Question: {question}

Instructions:
- Be specific and detailed
- Only use information from the resume
- If information isn't available, clearly state that
- Structure your response clearly
- Be professional and helpful

Answer:"""
            
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": sources,
                "relevance_scores": scores
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process question: {str(e)}")
            raise
    
    def health_check(self) -> dict:
        """Check system health"""
        try:
            # Test Pinecone
            pinecone_ok = False
            try:
                self.index.describe_index_stats()
                pinecone_ok = True
            except:
                pass
            
            # Test Groq
            groq_ok = False
            try:
                self.llm.invoke("test")
                groq_ok = True
            except:
                pass
            
            return {
                "pinecone_connected": pinecone_ok,
                "groq_connected": groq_ok
            }
        except:
            return {
                "pinecone_connected": False,
                "groq_connected": False
            }

# ============================================================================
# STEP 6: FastAPI Application
# ============================================================================
app = FastAPI(
    title="Resume RAG System",
    description="AI-powered resume analysis system using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = ResumeRAGSystem()
        logger.info("🚀 RAG System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        raise

def get_rag_system():
    """Dependency to get RAG system"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    return rag_system

# ============================================================================
# STEP 7: API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0; }
            input, textarea, button { width: 100%; padding: 10px; margin: 5px 0; }
            button { background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; }
            button:hover { background: #0056b3; }
            .result { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🤖 Resume RAG System</h1>
        <div class="container">
            <h3>1. Upload Resume</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" accept=".pdf" required>
                <input type="text" id="userId" placeholder="Enter User ID (optional)" value="dev_user">
                <button type="submit">Upload Resume</button>
            </form>
            <div id="uploadResult" class="result" style="display:none;"></div>
        </div>
        
        <div class="container">
            <h3>2. Ask Questions</h3>
            <form id="questionForm">
                <textarea id="question" placeholder="Ask about your resume..." rows="3"></textarea>
                <input type="text" id="questionUserId" placeholder="User ID" value="dev_user">
                <button type="submit">Ask Question</button>
            </form>
            <div id="questionResult" class="result" style="display:none;"></div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('fileInput').files[0]);
                formData.append('user_id', document.getElementById('userId').value);
                
                try {
                    const response = await fetch('/upload-resume', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('uploadResult').style.display = 'block';
                    document.getElementById('uploadResult').innerHTML = JSON.stringify(result, null, 2);
                } catch (error) {
                    document.getElementById('uploadResult').style.display = 'block';
                    document.getElementById('uploadResult').innerHTML = 'Error: ' + error.message;
                }
            };

            document.getElementById('questionForm').onsubmit = async function(e) {
                e.preventDefault();
                const data = {
                    question: document.getElementById('question').value,
                    user_id: document.getElementById('questionUserId').value
                };
                
                try {
                    const response = await fetch('/ask-question', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    document.getElementById('questionResult').style.display = 'block';
                    document.getElementById('questionResult').innerHTML = '<strong>Answer:</strong><br>' + result.answer;
                } catch (error) {
                    document.getElementById('questionResult').style.display = 'block';
                    document.getElementById('questionResult').innerHTML = 'Error: ' + error.message;
                }
            };
        </script>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check(rag: ResumeRAGSystem = Depends(get_rag_system)):
    """Health check endpoint"""
    health_status = rag.health_check()
    return HealthResponse(
        status="healthy" if all(health_status.values()) else "degraded",
        **health_status
    )

@app.post("/upload-resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
    rag: ResumeRAGSystem = Depends(get_rag_system)
):
    """Upload and process resume PDF"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    # Save temporarily and process
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the PDF
        chunks_created = rag.process_pdf(tmp_file_path, user_id)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return UploadResponse(
            message="Resume uploaded and processed successfully",
            user_id=user_id,
            chunks_created=chunks_created,
            file_name=file.filename
        )
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag: ResumeRAGSystem = Depends(get_rag_system)
):
    """Ask a question about the resume"""
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag.ask_question(request.question, request.user_id)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.delete("/delete-user-data/{user_id}")
async def delete_user_data(
    user_id: str,
    rag: ResumeRAGSystem = Depends(get_rag_system)
):
    """Delete all data for a user (GDPR compliance)"""
    try:
        # This would require implementing deletion logic
        # For now, return a placeholder response
        return {"message": f"Data deletion requested for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user data: {str(e)}")
