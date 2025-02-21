from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI

# Import dei blueprint
from routes.chatbot import chatbot
from routes.learning import learning
from routes.auth import auth_bp, login_required, verify_token_and_get_user
from routes.ml_routes import ml
from routes.image_routes import image
from routes.profile_routes import profile

# Initialize Firebase only once at module level
_firebase_initialized = False

def initialize_firebase():
    global _firebase_initialized
    if not _firebase_initialized:
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        _firebase_initialized = True
    return client

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Initialize Firebase and OpenAI
client = initialize_firebase()

# Assicurati che le cartelle necessarie esistano
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'profile_pictures'), exist_ok=True)

# Registrazione dei blueprint
app.register_blueprint(auth_bp, url_prefix='')
app.register_blueprint(chatbot, url_prefix='')
app.register_blueprint(learning, url_prefix='')
app.register_blueprint(ml, url_prefix='/ml')
app.register_blueprint(image, url_prefix='/image')
app.register_blueprint(profile, url_prefix='/profile')

@app.route('/')
def home():
    # Controlla se l'utente è autenticato
    user = verify_token_and_get_user()
    
    # Mostra sempre la landing page
    return render_template('landing.html',
                         firebase_api_key=os.getenv('FIREBASE_API_KEY'),
                         firebase_auth_domain=os.getenv('FIREBASE_AUTH_DOMAIN'),
                         firebase_project_id=os.getenv('FIREBASE_PROJECT_ID'),
                         firebase_app_id=os.getenv('FIREBASE_APP_ID'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
