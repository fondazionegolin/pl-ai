#!/usr/bin/env python3
"""
Script di utilità per riparare il database dell'utente.
Esegui questo script per creare o riparare il database dell'utente.
"""

import os
import sqlite3
import sys
from pathlib import Path

def get_user_db_path(username):
    """Ottiene il percorso del database per un utente specifico."""
    # Crea la directory per i database degli utenti se non esiste
    user_db_dir = Path("user_data")
    user_db_dir.mkdir(exist_ok=True)
    
    # Percorso del database dell'utente
    db_path = user_db_dir / f"{username}.db"
    return str(db_path.absolute())

def repair_user_db(username):
    """Ripara il database dell'utente."""
    print(f"Riparazione database per l'utente: {username}")
    
    # Ottieni il percorso del database
    db_path = get_user_db_path(username)
    print(f"Percorso database: {db_path}")
    
    # Verifica se il file esiste
    if not os.path.exists(db_path):
        print(f"Il database non esiste, verrà creato: {db_path}")
    
    # Connetti al database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Crea la tabella learning_units se non esiste
    print("Creazione tabella learning_units...")
    c.execute('''
    CREATE TABLE IF NOT EXISTS learning_units (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        quiz TEXT,
        answers TEXT DEFAULT '{}',
        total_questions INTEGER DEFAULT 0,
        correct_answers INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Verifica la struttura della tabella
    c.execute("PRAGMA table_info(learning_units)")
    columns = c.fetchall()
    print(f"Struttura tabella learning_units: {columns}")
    
    # Verifica l'integrità del database
    c.execute("PRAGMA integrity_check")
    integrity = c.fetchone()
    print(f"Controllo integrità: {integrity}")
    
    # Verifica se ci sono lezioni
    c.execute("SELECT COUNT(*) FROM learning_units")
    count = c.fetchone()[0]
    print(f"Numero di lezioni nel database: {count}")
    
    # Elenca tutte le lezioni
    if count > 0:
        print("Lezioni trovate:")
        c.execute("SELECT id, title FROM learning_units")
        for row in c.fetchall():
            print(f"  ID: {row[0]}, Titolo: {row[1]}")
    
    conn.commit()
    conn.close()
    
    print("Riparazione database completata con successo!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Inserisci il nome utente: ")
    
    repair_user_db(username)
