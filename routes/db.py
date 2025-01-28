import os
import sqlite3

def get_db():
    """Connette al database SQLite."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')
    return sqlite3.connect(db_path)
