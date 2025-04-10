# ==========================
# db_manager.py
# ==========================
# Database management and connection handling for the Quant-Aero-Log framework.

from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from config_env import DB_CONFIG, ENV
from utils.logger import get_logger, log_error

logger = get_logger("database")

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self) -> None:
        """Initialize the database manager with connection pooling."""
        self.engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}",
            poolclass=QueuePool,
            pool_size=DB_CONFIG['pool_size'],
            max_overflow=DB_CONFIG['max_overflow'],
            pool_timeout=30,
            pool_recycle=1800
        )
        
        # Configure session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Add event listeners
        self._setup_event_listeners()
        
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for connection management."""
        @event.listens_for(self.engine, "connect")
        def connect(dbapi_connection, connection_record):
            logger.info("database_connection_established")
            
        @event.listens_for(self.engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            logger.debug("database_connection_checked_out")
            
        @event.listens_for(self.engine, "checkin")
        def checkin(dbapi_connection, connection_record):
            logger.debug("database_connection_checked_in")
            
        @event.listens_for(self.engine, "reset")
        def reset(dbapi_connection, connection_record):
            logger.debug("database_connection_reset")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with proper cleanup."""
        session: Optional[Session] = None
        try:
            session = self.SessionLocal()
            yield session
            session.commit()
        except Exception as e:
            if session:
                session.rollback()
            log_error(e, {"component": "database", "operation": "session_management"})
            raise
        finally:
            if session:
                session.close()
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            log_error(e, {"component": "database", "operation": "connection_test"})
            return False
    
    def get_connection_count(self) -> int:
        """Get the current number of active connections."""
        return self.engine.pool.checkedin() + self.engine.pool.checkedout()
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        self.engine.dispose()
        logger.info("database_connections_closed")

# Global database manager instance
db_manager = DatabaseManager()

# Convenience function for getting a session
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session from the global manager."""
    with db_manager.get_session() as session:
        yield session 