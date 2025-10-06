"""
Session Manager for BioKnowdes
Handles session storage both locally (file-based) and on Heroku (PostgreSQL)
"""

import os
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# PostgreSQL support (optional for local development)
try:
    import sqlalchemy
    from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ PostgreSQL dependencies not available. Using file-based sessions.")

# SQLAlchemy setup
if POSTGRES_AVAILABLE:
    Base = declarative_base()

    class SessionData(Base):
        __tablename__ = 'sessions'
        
        session_id = Column(String(36), primary_key=True)
        documents = Column(Text)  # JSON string
        settings = Column(Text)   # JSON string
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SessionManager:
    """Unified session manager that works both locally and on Heroku."""
    
    def __init__(self):
        self.use_postgres = False
        self.db_session = None
        
        # Try to initialize PostgreSQL (local or Heroku)
        database_url = os.getenv('DATABASE_URL')
        if database_url and POSTGRES_AVAILABLE:
            try:
                print(f"🔗 [SESSION] Connecting to PostgreSQL: {database_url[:50]}...")
                
                # Fix for Heroku PostgreSQL URL
                if database_url.startswith('postgres://'):
                    database_url = database_url.replace('postgres://', 'postgresql://', 1)
                
                # Create engine with explicit connection settings
                self.engine = create_engine(
                    database_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    echo=False  # Set to True for SQL debugging
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    print("🔗 [SESSION] PostgreSQL connection test successful")
                
                # Create tables
                Base.metadata.create_all(self.engine)
                print("📋 [SESSION] Database tables created/verified")
                
                # Create session
                SessionLocal = sessionmaker(bind=self.engine)
                self.db_session = SessionLocal()
                self.use_postgres = True
                print("✅ [SESSION] PostgreSQL session storage initialized successfully")
                
            except Exception as e:
                print(f"❌ [SESSION] PostgreSQL init failed: {e}")
                print("📁 [SESSION] Falling back to file storage")
                self.use_postgres = False
        else:
            if not database_url:
                print("⚠️ [SESSION] No DATABASE_URL found in environment")
            if not POSTGRES_AVAILABLE:
                print("⚠️ [SESSION] PostgreSQL dependencies not available")
            print("📁 [SESSION] Using file-based storage")
        
        # Ensure local sessions directory exists (fallback only)
        if not self.use_postgres:
            os.makedirs('sessions', exist_ok=True)
            print("📁 [SESSION] File-based session storage initialized (fallback)")
    
    def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get documents for a session."""
        if self.use_postgres:
            return self._get_documents_postgres(session_id)
        else:
            return self._get_documents_file(session_id)
    
    def save_session_documents(self, session_id: str, documents: List[Dict]):
        """Save documents for a session."""
        if self.use_postgres:
            self._save_documents_postgres(session_id, documents)
        else:
            self._save_documents_file(session_id, documents)
    
    def get_session_settings(self, session_id: str) -> Dict:
        """Get settings for a session."""
        if self.use_postgres:
            return self._get_settings_postgres(session_id)
        else:
            return self._get_settings_file(session_id)
    
    def save_session_settings(self, session_id: str, settings: Dict):
        """Save settings for a session."""
        if self.use_postgres:
            self._save_settings_postgres(session_id, settings)
        else:
            self._save_settings_file(session_id, settings)
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        if self.use_postgres:
            self._clear_session_postgres(session_id)
        else:
            self._clear_session_file(session_id)
    
    # PostgreSQL implementations
    def _get_documents_postgres(self, session_id: str) -> List[Dict]:
        try:
            print(f"🔍 [POSTGRES] Looking for documents for session: {session_id}")
            session_data = self.db_session.query(SessionData).filter_by(session_id=session_id).first()
            if session_data and session_data.documents:
                documents = json.loads(session_data.documents)
                print(f"📄 [POSTGRES] Found {len(documents)} documents")
                return documents
            else:
                print(f"⚠️ [POSTGRES] No documents found for session: {session_id}")
                return []
        except Exception as e:
            print(f"❌ [POSTGRES] Error getting documents: {e}")
            return []
    
    def _save_documents_postgres(self, session_id: str, documents: List[Dict]):
        try:
            print(f"💾 [POSTGRES] Saving {len(documents)} documents for session: {session_id}")
            session_data = self.db_session.query(SessionData).filter_by(session_id=session_id).first()
            if session_data:
                print(f"📝 [POSTGRES] Updating existing session record")
                session_data.documents = json.dumps(documents)
                session_data.updated_at = datetime.utcnow()
            else:
                print(f"🆕 [POSTGRES] Creating new session record")
                session_data = SessionData(
                    session_id=session_id,
                    documents=json.dumps(documents),
                    settings=json.dumps({})
                )
                self.db_session.add(session_data)
            self.db_session.commit()
            print(f"✅ [POSTGRES] Successfully saved documents")
        except Exception as e:
            print(f"❌ [POSTGRES] Error saving documents: {e}")
            self.db_session.rollback()
    
    def _get_settings_postgres(self, session_id: str) -> Dict:
        try:
            session_data = self.db_session.query(SessionData).filter_by(session_id=session_id).first()
            if session_data and session_data.settings:
                return json.loads(session_data.settings)
            return {}
        except Exception as e:
            print(f"Error getting settings from PostgreSQL: {e}")
            return {}
    
    def _save_settings_postgres(self, session_id: str, settings: Dict):
        try:
            session_data = self.db_session.query(SessionData).filter_by(session_id=session_id).first()
            if session_data:
                session_data.settings = json.dumps(settings)
                session_data.updated_at = datetime.utcnow()
            else:
                session_data = SessionData(
                    session_id=session_id,
                    documents=json.dumps([]),
                    settings=json.dumps(settings)
                )
                self.db_session.add(session_data)
            self.db_session.commit()
        except Exception as e:
            print(f"Error saving settings to PostgreSQL: {e}")
            self.db_session.rollback()
    
    def _clear_session_postgres(self, session_id: str):
        try:
            session_data = self.db_session.query(SessionData).filter_by(session_id=session_id).first()
            if session_data:
                self.db_session.delete(session_data)
                self.db_session.commit()
        except Exception as e:
            print(f"Error clearing session from PostgreSQL: {e}")
            self.db_session.rollback()
    
    # File-based implementations (local development)
    def _get_documents_file(self, session_id: str) -> List[Dict]:
        session_file = f'sessions/{session_id}.json'
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                return session_data.get('documents', [])
            except:
                return []
        return []
    
    def _save_documents_file(self, session_id: str, documents: List[Dict]):
        session_file = f'sessions/{session_id}.json'
        existing_data = {'documents': [], 'settings': {}}
        
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        existing_data['documents'] = documents
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False)
    
    def _get_settings_file(self, session_id: str) -> Dict:
        settings_file = f'sessions/{session_id}_settings.json'
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_settings_file(self, session_id: str, settings: Dict):
        settings_file = f'sessions/{session_id}_settings.json'
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False)
    
    def _clear_session_file(self, session_id: str):
        session_file = f'sessions/{session_id}.json'
        settings_file = f'sessions/{session_id}_settings.json'
        
        for file_path in [session_file, settings_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

# Global session manager instance  
# Initialize after load_dotenv() is called in app.py
session_manager = None

def get_session_manager():
    """Get or create the global session manager instance."""
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager