import os
from typing import AsyncGenerator, Dict, Optional, Union
import chainlit as cl
import chromadb
import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from autogen import Agent, UserProxyAgent, AssistantAgent, ConversableAgent
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from autogen import register_function
from prompts.prompt_template import DOCUMENT_AGENT_PROMPT, USER_PROXY_AGENT_PROMPT
from tools.arxiv_search_tool import query_arxiv, query_web
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
    
    
# Define Chainlit-integrated agent classes
class ChainlitDocumentAssistantAgent(AssistantAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f'{message}',
                author="DocumentAnalystAgent",
            ).send()
        )
        return super(ChainlitDocumentAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitDocumentUserProxyAgent(UserProxyAgent):
    async def get_human_input(self, prompt: str) -> str:
        """Get human input via Chainlit UI"""
        async def ask_helper(func, **kwargs):
            res = await func(**kwargs).send()
            while not res:
                res = await func(**kwargs).send()
            return res
            
        if prompt.startswith(
            "Please give feedback to DocumentAnalystAgent. Press enter to skip and use auto-reply"
        ):
            res = await ask_helper(
                cl.AskActionMessage,
                content="Continue or provide feedback?",
                actions=[
                    cl.Action(
                        name="continue",
                        payload={"value": "continue"},
                        label="✅ Continue",
                    ),
                    cl.Action(
                        name="feedback",
                        payload={"value": "feedback"},
                        label="💬 Provide feedback",
                    ),
                    cl.Action(
                        name="exit",
                        payload={"value": "exit"},
                        label="🔚 Exit Conversation",
                    ),
                ],
            )
            if res.get("payload").get("value") == "continue":
                return ""
            if res.get("payload").get("value") == "exit":
                return "exit"
                
        reply = await ask_helper(cl.AskUserMessage, content=prompt, timeout=60)
        return reply["content"].strip()
        
    # def send(
    #     self,
    #     message: Union[Dict, str],
    #     recipient: Agent,
    #     request_reply: Optional[bool] = None,
    #     silent: Optional[bool] = False,
    # ):
    #     cl.run_sync(
    #         cl.Message(
    #             content=f'{message}',
    #             author="User (Feedback)",
    #         ).send()
    #     )
    #     return super(ChainlitDocumentUserProxyAgent, self).send(
    #         message=message,
    #         recipient=recipient,
    #         request_reply=request_reply,
    #         silent=silent,
    #     )

class DocumentQAAgent:
    def __init__(self):
        # Model client
        self.client = AzureAIChatCompletionClient(
            model="gpt-4",
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
                "model": "gpt-4",
                "api_key": os.getenv("GITHUB_TOKEN"),
                "base_url": os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.inference.ai.azure.com"),
            }],
        }
        self.assistant = self._create_doc_assistant(self.llm_config)
    
    def _create_doc_assistant(self, llm_config):
        """Create the AutoGen assistant with the proper configuration"""
        assistant = ChainlitDocumentAssistantAgent(
            name="DocumentAnalystAgent",
            llm_config=llm_config,
            system_message=DOCUMENT_AGENT_PROMPT,
        )
        return assistant
        
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
        user_proxy = ChainlitDocumentUserProxyAgent(
            name="User",
            llm_config=False,  # This disables LLM usage for this agent
            human_input_mode="TERMINATE",  # Change this to ALWAYS to get input from a real user
            max_consecutive_auto_reply=1,  # Set to 0 to require human input for every reply
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        register_function(
            query_web,
            caller=self.assistant, 
            executor=user_proxy, 
            name="web_search",  
            description="Searches the web for relevant academic content",
        )
        
        yield "⏳ Thinking..."
        
        try:
            chat_result = user_proxy.initiate_chat(
                self.assistant,
                message=self._get_user_proxy_prompt(question, context),
            )
            assistant_messages = [msg for msg in chat_result.chat_history if msg["sender"] == "DocumentAnalystAgent"]
            if not assistant_messages:
                yield "Failed to generate a response."
                
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        yield " "
    
    def cleanup(self):
        """Clean up the temporary collection"""
        if self.vector_store:
            # Delete the collection (for Chroma)
            self.chroma_client.delete_collection(self.collection_name)