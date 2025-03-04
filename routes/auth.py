from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Crea il Blueprint per l'autenticazione
auth_bp = Blueprint('auth', __name__)

# Inizializzazione Firebase Admin SDK
cred = credentials.Certificate('firebase-credentials.json')

# Inizializza Firebase Admin e Firestore
try:
    # Se l'app è già inizializzata, usa quella esistente
    default_app = firebase_admin.get_app()
except ValueError:
    # Altrimenti inizializza una nuova app
    default_app = firebase_admin.initialize_app(cred)

# Inizializza il client Firestore
db = firestore.client()
print("Firebase Admin SDK e Firestore inizializzati con successo")

# Verifica che Firestore sia accessibile
try:
    # Prova ad accedere alla collection users
    users_ref = db.collection('users')
    print("Connessione a Firestore verificata con successo")
except Exception as e:
    print(f"Errore nell'accesso a Firestore: {str(e)}")
    raise

auth_bp = Blueprint('auth', __name__)

def extract_token():
    # Prima controlla l'header Authorization
    auth_header = request.headers.get('Authorization')
    print(f'[DEBUG] Auth header: {auth_header}')
    
    if auth_header:
        try:
            # Il token dovrebbe essere nel formato "Bearer <token>"
            token = auth_header.split(' ')[1]
            print(f'[DEBUG] Token from header: {token[:20]}...')
            return token
        except (IndexError, AttributeError):
            print('[DEBUG] Failed to extract token from header')
            pass
    
    # Se non c'è token nell'header, controlla nella sessione
    session_token = session.get('auth_token')
    print(f'[DEBUG] Session token: {session_token[:20] if session_token else None}...')
    if session_token:
        return session_token
        
    print('[DEBUG] No token found')
    return None

def verify_token_and_get_user():
    token = extract_token()
    if not token:
        pass
        return None
        
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        pass
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print(f'\n[DEBUG] --- login_required called for endpoint: {request.endpoint} ---')
        
        # Route che non richiedono autenticazione
        public_routes = ['home', 'landing', 'auth.login', 'auth.verify_token']
        if request.endpoint in public_routes:
            print(f'[DEBUG] Skipping auth for public route: {request.endpoint}')
            return f(*args, **kwargs)
        
        # Ottieni il token (prima dall'header, poi dalla sessione)
        token = extract_token()
        print(f'[DEBUG] Extracted token: {token[:20] if token else None}...')
        
        if not token:
            print('[DEBUG] No token found, clearing session')
            session.clear()
            # Se è una richiesta API, restituisci 401
            if request.headers.get('Accept', '').startswith('application/json') or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Unauthorized', 'redirect': url_for('home')}), 401
            return redirect(url_for('home'))
        
        try:
            # Verifica il token
            print('[DEBUG] Verifying token with Firebase')
            decoded_token = auth.verify_id_token(token)
            
            # Salva il token e le info utente nella sessione
            print('[DEBUG] Token verified, updating session')
            session['auth_token'] = token
            session['user_uid'] = decoded_token['uid']
            session['user_email'] = decoded_token.get('email', '')
            
            # Verifica se l'utente esiste già in Firestore
            db = firestore.client()
            user_ref = db.collection('users').document(decoded_token['uid'])
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                # Se l'utente non esiste, crealo
                print(f'[DEBUG] Creating new user in Firestore: {decoded_token["uid"]}')
                
                # Estrai il nome e cognome dall'email o dal display name
                name_parts = decoded_token.get('name', '').split() if decoded_token.get('name') else ['', '']
                first_name = name_parts[0] if name_parts else ''
                last_name = name_parts[1] if len(name_parts) > 1 else ''
                
                user_data = {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': decoded_token.get('email', ''),
                    'bio': '',
                    'profile_picture': decoded_token.get('picture', ''),
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP
                }
                
                user_ref.set(user_data)
                print('[DEBUG] New user created in Firestore')
            else:
                print(f'[DEBUG] User already exists in Firestore: {decoded_token["uid"]}')
            
            # Aggiungi le info utente alla richiesta
            request.user = decoded_token
            
            print('[DEBUG] Auth successful, proceeding to route')
            return f(*args, **kwargs)
            
        except Exception as e:
            print(f'[DEBUG] Auth failed: {str(e)}')
            session.clear()
            # Se è una richiesta API, restituisci 401
            if request.headers.get('Accept', '').startswith('application/json') or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Unauthorized', 'redirect': url_for('home')}), 401
            return redirect(url_for('home'))
            
    return decorated_function
            
    return decorated_function

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    print('\n[DEBUG] --- verify_token called ---')
    current_token = extract_token()
    print(f'[DEBUG] Current token: {current_token[:20] if current_token else None}...')
    
    # Check session cache first
    session_token = session.get('auth_token')
    print(f'[DEBUG] Session token: {session_token[:20] if session_token else None}...')
    
    if current_token and current_token == session_token:
        print('[DEBUG] Token match found in session')
        return jsonify({
            'uid': session.get('user_uid'),
            'email': session.get('user_email'),
            'name': session.get('user_name')
        }), 200

    # Verify token if not in cache
    print('[DEBUG] Verifying token with Firebase')
    user = verify_token_and_get_user()
    if not user:
        print('[DEBUG] Token verification failed')
        return jsonify({'error': 'Invalid token'}), 401

    # Update session cache
    print('[DEBUG] Updating session with new token')
    session['auth_token'] = current_token
    session['user_uid'] = user['uid']
    session['user_email'] = user.get('email', '')
    session['user_name'] = user.get('name', '')
    
    print('[DEBUG] Token verification successful')
    return jsonify({
        'uid': user['uid'],
        'email': user.get('email', ''),
        'name': user.get('name', '')
    }), 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Endpoint per il logout dell'utente."""
    # Pulisci la sessione
    session.clear()
    return jsonify({'message': 'Logout effettuato con successo'}), 200

@auth_bp.route('/check-auth', methods=['GET'])
def check_auth():
    """Endpoint per verificare lo stato dell'autenticazione."""
    user = verify_token_and_get_user()
    
    if not user:
        return jsonify({'authenticated': False}), 401
        
    return jsonify({
        'authenticated': True,
        'user': {
            'uid': user['uid'],
            'email': user.get('email', ''),
            'name': user.get('name', '')
        }
    }), 200

@auth_bp.route('/login')
def login():
    return render_template('landing.html',
                         firebase_api_key=os.getenv('FIREBASE_API_KEY'),
                         firebase_auth_domain=os.getenv('FIREBASE_AUTH_DOMAIN'),
                         firebase_project_id=os.getenv('FIREBASE_PROJECT_ID'),
                         firebase_app_id=os.getenv('FIREBASE_APP_ID'))

@auth_bp.route('/main-console')
@login_required
def main_console():
    return render_template('main-console.html',
                         firebase_api_key=os.getenv('FIREBASE_API_KEY'),
                         firebase_auth_domain=os.getenv('FIREBASE_AUTH_DOMAIN'),
                         firebase_project_id=os.getenv('FIREBASE_PROJECT_ID'),
                         firebase_app_id=os.getenv('FIREBASE_APP_ID'))

# Redirecting old dashboard URL to new main-console URL for backward compatibility
@auth_bp.route('/dashboard')
@login_required
def dashboard():
    return redirect(url_for('auth.main_console'))

# Redirecting console URL to main-console for consistency
@auth_bp.route('/console')
@login_required
def console():
    return redirect(url_for('auth.main_console'))

from datetime import datetime

def get_current_user():
    """Recupera le informazioni dell'utente corrente dal database."""
    try:
        # Verifica il token e ottieni l'ID utente
        token = extract_token()
        if not token:
            return None
            
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        # Recupera i dati dell'utente da Firestore
        db = firestore.client()
        user_doc = db.collection('users').document(uid).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_data['uid'] = uid
            return user_data
        else:
            # Se il documento non esiste, crea un nuovo profilo utente
            now = datetime.now().isoformat()
            user_data = {
                'uid': uid,
                'email': decoded_token.get('email', ''),
                'first_name': '',
                'last_name': '',
                'bio': '',
                'profile_picture': '',
                'created_at': now,
                'updated_at': now
            }
            db.collection('users').document(uid).set(user_data)
            return user_data
            
    except Exception as e:
        print(f"Errore nel recupero dell'utente: {str(e)}")
        return None

def update_user_profile(uid, data):
    """Aggiorna le informazioni del profilo utente nel database."""
    try:
        # Validazione dei campi
        allowed_fields = {'bio', 'email', 'first_name', 'last_name', 'profile_picture'}
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        # Aggiungi il timestamp di aggiornamento
        update_data['updated_at'] = datetime.now().isoformat()
        
        # Validazione dei tipi
        for field, value in update_data.items():
            if field in ['bio', 'email', 'first_name', 'last_name', 'profile_picture']:
                if not isinstance(value, str):
                    raise ValueError(f"Il campo {field} deve essere una stringa")
                update_data[field] = str(value)
        
        db = firestore.client()
        db.collection('users').document(uid).update(update_data)
        return True
    except Exception as e:
        print(f"Errore nell'aggiornamento del profilo: {str(e)}")
        return False

def allowed_file(filename, allowed_extensions):
    """Verifica se l'estensione del file è consentita."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
