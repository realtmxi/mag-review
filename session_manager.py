import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import chainlit as cl
import os
import json

class AgentType(Enum):
    LITERATURE = "literature"
    PAPER_REVIEW = "paper_review"
    QA = "qa"

class SessionStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"

class Session:
    """Represents a chat session with metadata and history references"""
    def __init__(
        self, 
        session_id: str, 
    ):
        self.session_id = session_id
        self.context_data: Dict[str, Any] = {}
        self.chat_history: list[Dict[str, str]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary representation"""
        return {
            "session_id": self.session_id,
            "context_data": self.context_data,
            "chat_history": self.chat_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session object from dictionary"""
        session = cls(session_id=data["session_id"])
        session.context_data = data.get("context_data", {})
        session.chat_history = data.get("chat_history", [])
        return session
    
    def update_context(self, key: str, value: Any)-> None:
        """Add or update data in the context. """
        self.context_data[key] = value
    
    def get_context(self, key: str) -> Any:
        """
        Retrieve context data by key.
        """
        return self.context_data.get(key)

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the session's chat history.
        """
        self.chat_history.append({"role": role, "content": content})


class SessionManager:
    """Manages research sessions with in-memory and file-based persistence"""
    
    def __init__(self, storage_dir: str = "sessions"):
        """
        Initialize the session manager with storage location
        
        Args:
            storage_dir: Directory to store session files
        """
        self.sessions: Dict[str, 'Session'] = {}  # In-memory cache
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        # Load existing sessions from disk
        self._load_sessions()
    
    def _load_sessions(self) -> None:
        """Load all session files from storage"""
        if not os.path.exists(self.storage_dir):
            return
            
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    session_id = filename.replace('.json', '')
                    session_path = os.path.join(self.storage_dir, filename)
                    
                    with open(session_path, 'r') as f:
                        session_data = json.load(f)
                        
                    session = Session.from_dict(session_data)
                    self.sessions[session_id] = session
                except Exception as e:
                    print(f"Error loading session {filename}: {str(e)}")
    
    def _save_session(self, session: 'Session') -> None:
        """Save session to disk"""
        session_path = os.path.join(self.storage_dir, f"{session.session_id}.json")
        
        with open(session_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)
    
    def create_session(self) -> 'Session':
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        
        # Store in memory
        self.sessions[session_id] = session
        
        # Save to disk
        self._save_session(session)
        
        return session
    
    def get_session(self, session_id: str) -> Optional['Session']:
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List['Session']:
        """Get all sessions"""
        return list(self.sessions.values())
    
    def update_session(self, session: 'Session') -> None:
        """Update a session"""
        # Update in memory
        self.sessions[session.session_id] = session
        
        # Save to disk
        self._save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id not in self.sessions:
            return False
            
        # Remove from memory
        del self.sessions[session_id]
        
        # Remove from disk
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            os.remove(session_path)
            
        return True
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to a session"""
        session = self.get_session(session_id)
        if not session:
            return False
            
        session.add_message(role, content)
        self.update_session(session)
        return True
    
    def update_context(self, session_id: str, key: str, value: Any) -> bool:
        """Update context data in a session"""
        session = self.get_session(session_id)
        if not session:
            return False
            
        session.update_context(key, value)
        self.update_session(session)
        return True
    
    def get_context(self, session_id: str, key: str) -> Optional[Any]:
        """Get context data from a session"""
        session = self.get_session(session_id)
        if not session:
            return None
            
        return session.get_context(key)
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get all messages in a session"""
        session = self.get_session(session_id)
        if not session:
            return []
            
        return session.chat_history