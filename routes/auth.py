from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth
import os
from dotenv import load_dotenv

load_dotenv()

# Inizializzazione Firebase Admin SDK
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": os.getenv('FIREBASE_PROJECT_ID'),
    "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
    "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
    "client_id": os.getenv('FIREBASE_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
})

# Inizializza Firebase Admin
try:
    default_app = firebase_admin.initialize_app(cred)
except ValueError:
    default_app = firebase_admin.get_app()

auth_bp = Blueprint('auth', __name__)

def extract_token():
    auth_header = request.headers.get('Authorization')
    print("Auth header:", auth_header)  # Debug
    
    if not auth_header:
        print("Nessun token trovato nell'header")  # Debug
        return None
        
    try:
        # Il token dovrebbe essere nel formato "Bearer <token>"
        token = auth_header.split(' ')[1]
        print("Token estratto:", token[:20] + "...")  # Debug
        return token
    except (IndexError, AttributeError):
        print("Formato token non valido")  # Debug
        return None

def verify_token_and_get_user():
    token = extract_token()
    if not token:
        print("Token non fornito")  # Debug
        return None
        
    try:
        print("Tentativo di verifica token...")  # Debug
        decoded_token = auth.verify_id_token(token)
        print("Token verificato con successo:", decoded_token)  # Debug
        return decoded_token
    except Exception as e:
        print("Errore verifica token:", str(e))  # Debug
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("Checking authentication...")  # Debug
        
        # Controlla prima l'header Authorization
        auth_token = extract_token()
        if auth_token:
            try:
                decoded_token = auth.verify_id_token(auth_token)
                request.user = decoded_token
                session['auth_token'] = auth_token  # Aggiorna il token nella sessione
                return f(*args, **kwargs)
            except Exception as e:
                print(f"Token verification failed: {str(e)}")
                session.pop('auth_token', None)  # Rimuovi il token non valido dalla sessione
        
        # Se non c'è un token valido nell'header, prova con la sessione
        session_token = session.get('auth_token')
        if session_token:
            try:
                decoded_token = auth.verify_id_token(session_token)
                request.user = decoded_token
                return f(*args, **kwargs)
            except Exception as e:
                print(f"Session token verification failed: {str(e)}")
                session.pop('auth_token', None)  # Rimuovi il token non valido
        
        print("Authentication failed")  # Debug
        # Pulisci la sessione
        session.clear()
        
        if request.headers.get('Accept', '').startswith('application/json'):
            return jsonify({'error': 'Unauthorized', 'redirect': url_for('home')}), 401
        return redirect(url_for('home'))
            
    return decorated_function

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    print("Richiesta /verify-token ricevuta")  # Debug
    print("Headers:", dict(request.headers))  # Debug
    
    user = verify_token_and_get_user()
    if not user:
        print("Verifica token fallita")  # Debug
        return jsonify({'error': 'Invalid token'}), 401
    
    print("Utente verificato:", user)  # Debug
    session['auth_token'] = extract_token()  # Salva il token nella sessione
    return jsonify({
        'uid': user['uid'],
        'email': user.get('email', ''),
        'name': user.get('name', '')
    }), 200

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

@auth_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html',
                         firebase_api_key=os.getenv('FIREBASE_API_KEY'),
                         firebase_auth_domain=os.getenv('FIREBASE_AUTH_DOMAIN'),
                         firebase_project_id=os.getenv('FIREBASE_PROJECT_ID'),
                         firebase_app_id=os.getenv('FIREBASE_APP_ID'))
