import os
from typing import List, Dict, Any, AsyncGenerator
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from prompts.prompt_template import DOCUMENT_AGENT_PROMPT, USER_PROXY_AGENT_PROMPT
from dotenv import load_dotenv
load_dotenv()

# embedding model cache
cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

# Singleton Embeddings Manager to load embedding model only once
class EmbeddingsManager:
    _instance = None  
    _embeddings = None  
    
    @classmethod
    def get_embeddings(cls):
        """Return the existing embeddings instance or create a new one if none exists"""
        if cls._embeddings is None:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
            cls._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                  'device': 'cpu', # 'cuda' if GPU
                }  
            )
        return cls._embeddings

class DocumentQAAgent:
    def __init__(self):
        # Model client
        self.client = AzureAIChatCompletionClient(
            model="gpt-4o",
            endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com"),
            credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
                model_info={
                    "json_output": True,
                    "function_calling": True,
                    "vision": False,
                    "family": "unknown",
                    "structured_output": True
                },
        )
        
        # Embedding
        self.embeddings = EmbeddingsManager.get_embeddings()
        
        # Chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # In-memory vectore store
        self.chroma_client = chromadb.Client()
        self.collection_name = f"temp_collection_{os.urandom(4).hex()}"
        self.vector_store = None
        
        # Assistant
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("GITHUB_TOKEN"),
                "base_url": os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com"),
            }]
        }
        self.assistant = self._create_doc_assistant(self.llm_config)
    
    def _create_doc_assistant(self, llm_config):
        """Create the AutoGen assistant with the proper configuration"""
        return AssistantAgent(
            name="DocumentAnalystAgent",
            llm_config=llm_config,
            system_message=DOCUMENT_AGENT_PROMPT
        )
      
    def _retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant document sections for a given query"""
        if not self.vector_store:
            return "No documents have been processed yet."
        
        results = self.vector_store.similarity_search(query, k=top_k)
        
        context_sections = []
        
        # Group results by source file
        source_groups = {}
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Format sections by source file
        for source, docs in source_groups.items():
            # Create header for this source file
            source_header = f"=== From document: {source} ===\n"
            sections = [source_header]
            
            # Add each section with appropriate page info
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", "N/A")
                
                # Format page information appropriately based on file type
                if source.lower().endswith(('.pdf')):
                    location_info = f"Page {page}"
                elif source.lower().endswith(('.csv')):
                    location_info = f"Entry {page}"
                else:
                    location_info = f"Section {i+1}"
                
                # Add formatted section with content and location
                sections.append(f"[{location_info}]\n{doc.page_content}\n")
            
            # Join all sections for this source
            context_sections.append("\n".join(sections))
        
        # Join all sources with clear separation
        return "\n\n" + "\n\n".join(context_sections)
    
    def _get_user_proxy_prompt(self, user_question, document_context):
        formatted_prompt = USER_PROXY_AGENT_PROMPT.format(
            question=user_question,
            context=document_context
        )
        return formatted_prompt
    
    def _load_documents(self, file_path: str, file_type: str, file_name: str):
        """Load documents based on file type"""
        if file_type == "pdf":
            # Loads one file into multiple documents (one per page)
            loader = PyMuPDFLoader(file_path)
        elif file_type in ["txt", "text"]:
            # Loads one file into one document
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_type in ["docx", "doc"]:
            # Loads one file into one document
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Load documents
        documents = loader.load()
        
        # Normalize metadata - filename/page
        for doc in documents:
            doc.metadata["source"] = file_name
            # For non-PDF files that don't have page numbers
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
        
        return documents
    
    def process_document(self, file_path: str, file_type: str, file_name: str) -> int:
        """Process a document and store it in the vector database"""
        # Load document with normalized metadata
        documents = self._load_documents(file_path, file_type, file_name)
        
        # Chunking
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store with embedded file content
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name
            )
        else:
            self.vector_store.add_documents(chunks)
        
        return len(chunks)
    
    async def answer_question(self, question: str) -> AsyncGenerator[str, None]:
        """Answer a question using the document context and stream the response"""
        # get top5 most relevant chunks
        context = self._retrieve_context(question, 5)
        
        if context == "No documents have been processed yet.":
            yield "Please upload documents first."
            return
        
        # Create user proxy for interaction
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER", #TODO: update later
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        response = ""
        try:
            chat_result = user_proxy.initiate_chat(
                self.assistant,
                message=self._get_user_proxy_prompt(question, context)
            )
            
            if chat_result.summary:
                response = chat_result.summary
            else:
                # Fallback to searching in chat history if summary is empty
                for message in reversed(chat_result.chat_history):
                    if message.get("name") == "DocumentAnalystAgent":
                        response = message["content"]
                        break
            if not response:
                response = "Failed to generate a response."
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        # Simulate streaming by yielding response in chunks
        chunk_size = 10  
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
    
    def cleanup(self):
        """Clean up the temporary collection"""
        if self.vector_store:
            # Delete the collection (for Chroma)
            self.chroma_client.delete_collection(self.collection_name)