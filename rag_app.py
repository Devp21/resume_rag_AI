pip install langchain langchain-groq chromadb sentence-transformers pypdf
pip install python-dotenv

# Simple Resume RAG System - Test in Colab
# Minimal version to test the core functionality

# ============================================================================
# STEP 1: Install packages
# ============================================================================
# ============================================================================
# STEP 2: Setup
# ============================================================================
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# STEP 3: Simple RAG function
# ============================================================================
def create_resume_rag(pdf_path):
    """Create RAG system from resume PDF"""
    
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # 3. Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Vector store created")
    
    # 4. Setup LLM
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1)
    
    return vectorstore, llm

def ask_question(vectorstore, llm, question):
    """Ask a question about the resume"""
    
    # Get relevant chunks
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""
    Based on the following resume content, answer the question.
    
    Resume Content:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    # Get response
    response = llm.invoke(prompt)
    return response.content

# ============================================================================
# STEP 4: Usage Example
# ============================================================================

# Upload your resume PDF to Colab first, then update the path below
PDF_PATH = "C:/Users/Hp/Desktop/AI/resume.pdf"  # Update this path

# Initialize the system (run this once)
print("Setting up RAG system...")
vectorstore, llm = create_resume_rag(PDF_PATH)
print("✅ System ready!")

# ============================================================================
# STEP 5: Ask questions
# ============================================================================

# Test questions - run these cells to test
questions = [
    "What programming languages are mentioned?",
    "Tell me about the work experience",
    "What education does this person have?",
    "What are the key skills?"
]

# Ask each question
for question in questions:
    print(f"\n🤔 Question: {question}")
    answer = ask_question(vectorstore, llm, question)
    print(f"💡 Answer: {answer}")
    print("-" * 80)

# ============================================================================
# STEP 6: Interactive questioning (optional)
# ============================================================================

# Uncomment below for interactive mode
"""
while True:
    question = input("\nAsk about your resume (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    answer = ask_question(vectorstore, llm, question)
    print(f"\nAnswer: {answer}")
"""

print("\n🎉 Test complete! If this works, we can add UI later.")
