import os
import sqlite3
import hashlib
import uuid
from functools import wraps
from flask import session, redirect, url_for, g

def get_db():
    """Connette al database SQLite principale."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_db(username):
    """Connette al database personalizzato di un utente."""
    user_db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_databases')
    os.makedirs(user_db_dir, exist_ok=True)
    db_path = os.path.join(user_db_dir, f'{username}.sqlite')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inizializza il database principale con le tabelle necessarie."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Crea la tabella degli utenti se non esiste
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        salt TEXT NOT NULL,
        firstname TEXT,
        lastname TEXT,
        email TEXT,
        avatar TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Verifica se le colonne aggiuntive esistono già, altrimenti le aggiunge
    try:
        # Controlla se la colonna firstname esiste
        cursor.execute("SELECT firstname FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila
        cursor.execute("ALTER TABLE users ADD COLUMN firstname TEXT")
        
    # Aggiungi colonne per il tracciamento dei crediti API
    try:
        # Controlla se la colonna stability_credits esiste
        cursor.execute("SELECT stability_credits FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila con valore predefinito 0
        cursor.execute("ALTER TABLE users ADD COLUMN stability_credits INTEGER DEFAULT 0")
    
    try:
        # Controlla se la colonna openai_credits esiste
        cursor.execute("SELECT openai_credits FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila con valore predefinito 0
        cursor.execute("ALTER TABLE users ADD COLUMN openai_credits INTEGER DEFAULT 0")
    
    try:
        # Controlla se la colonna lastname esiste
        cursor.execute("SELECT lastname FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila
        cursor.execute("ALTER TABLE users ADD COLUMN lastname TEXT")
    
    try:
        # Controlla se la colonna email esiste
        cursor.execute("SELECT email FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    
    try:
        # Controlla se la colonna avatar esiste
        cursor.execute("SELECT avatar FROM users LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila
        cursor.execute("ALTER TABLE users ADD COLUMN avatar TEXT")
    
    conn.commit()
    conn.close()

def init_user_db(username):
    """Inizializza il database personalizzato di un utente."""
    conn = get_user_db(username)
    cursor = conn.cursor()
    
    # Crea le tabelle per i dati dell'utente
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_type TEXT NOT NULL,
        data_name TEXT NOT NULL,
        data_value TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Crea un hash sicuro della password."""
    if salt is None:
        salt = uuid.uuid4().hex
    
    # Combina password e salt, poi crea l'hash
    hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed_password, salt

def register_user(username, password):
    """Registra un nuovo utente."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Verifica se l'utente esiste già
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone() is not None:
            conn.close()
            return False, "Username già in uso"
        
        # Crea l'hash della password
        hashed_password, salt = hash_password(password)
        
        # Inserisci il nuovo utente
        cursor.execute(
            "INSERT INTO users (username, password, salt) VALUES (?, ?, ?)",
            (username, hashed_password, salt)
        )
        conn.commit()
        
        # Inizializza il database personalizzato dell'utente
        init_user_db(username)
        
        return True, "Registrazione completata con successo"
    except Exception as e:
        conn.rollback()
        return False, f"Errore durante la registrazione: {str(e)}"
    finally:
        conn.close()

def authenticate_user(username, password):
    """Autentica un utente."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Cerca l'utente nel database
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user is None:
            return False, "Username o password non validi"
        
        # Verifica la password
        hashed_password, _ = hash_password(password, user['salt'])
        if hashed_password != user['password']:
            return False, "Username o password non validi"
        
        return True, "Autenticazione riuscita"
    except Exception as e:
        return False, f"Errore durante l'autenticazione: {str(e)}"
    finally:
        conn.close()

def login_required(f):
    """Decoratore per richiedere il login per accedere a certe route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def update_api_credits(username, api_type, amount=1):
    """Aggiorna i crediti API utilizzati dall'utente.
    
    Args:
        username (str): Nome utente
        api_type (str): Tipo di API ('stability' o 'openai')
        amount (int): Quantità di crediti da aggiungere (default: 1)
    """
    print(f"DEBUG - update_api_credits called with: username={username}, api_type={api_type}, amount={amount}")
    
    # Verifica che l'amount sia un numero intero positivo
    if not isinstance(amount, int) and not isinstance(amount, float):
        print(f"DEBUG - Invalid amount type: {type(amount)}")
        amount = 1
    
    # Converti in intero se necessario e assicurati che sia positivo
    amount = max(1, int(amount))  # Almeno 1 token per chiamata
    print(f"DEBUG - Final amount to add: {amount}")
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Prima verifichiamo i valori attuali
        cursor.execute(
            "SELECT stability_credits, openai_credits FROM users WHERE username = ?",
            (username,)
        )
        current_values = cursor.fetchone()
        if current_values:
            print(f"DEBUG - Current values: stability={current_values['stability_credits']}, openai={current_values['openai_credits']}")
        else:
            print(f"DEBUG - User {username} not found in database")
            return False
            
        if api_type == 'stability':
            # Assicurati che il valore non sia NULL prima di aggiornare
            cursor.execute(
                "UPDATE users SET stability_credits = COALESCE(stability_credits, 0) + ? WHERE username = ?",
                (amount, username)
            )
            print(f"DEBUG - Updated stability_credits for {username} by adding {amount}")
        elif api_type == 'openai':
            # Assicurati che il valore non sia NULL prima di aggiornare
            cursor.execute(
                "UPDATE users SET openai_credits = COALESCE(openai_credits, 0) + ? WHERE username = ?",
                (amount, username)
            )
            print(f"DEBUG - Updated openai_credits for {username} by adding {amount}")
        else:
            print(f"DEBUG - Unknown API type: {api_type}")
            return False
        
        # Verifichiamo i nuovi valori
        cursor.execute(
            "SELECT stability_credits, openai_credits FROM users WHERE username = ?",
            (username,)
        )
        new_values = cursor.fetchone()
        if new_values:
            print(f"DEBUG - New values: stability={new_values['stability_credits']}, openai={new_values['openai_credits']}")
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"DEBUG - Errore nell'aggiornamento dei crediti API: {str(e)}")
        return False
    finally:
        conn.close()

def get_api_credits(username):
    """Ottiene i crediti API utilizzati dall'utente.
    
    Args:
        username (str): Nome utente
        
    Returns:
        dict: Dizionario con i crediti per ogni tipo di API
    """
    print(f"DEBUG - get_api_credits called for user: {username}")
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT stability_credits, openai_credits FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()
        
        if user is None:
            print(f"DEBUG - User {username} not found in database")
            return {'stability': 0, 'openai': 0}
        
        # Verifica che i valori non siano None
        stability_credits = user['stability_credits'] if user['stability_credits'] is not None else 0
        openai_credits = user['openai_credits'] if user['openai_credits'] is not None else 0
        
        print(f"DEBUG - Retrieved API credits for {username}: stability={stability_credits}, openai={openai_credits}")
        
        return {
            'stability': stability_credits,
            'openai': openai_credits
        }
    except Exception as e:
        print(f"DEBUG - Errore nel recupero dei crediti API: {str(e)}")
        return {'stability': 0, 'openai': 0}
    finally:
        conn.close()

# Inizializza il database all'avvio dell'applicazione
init_db()
