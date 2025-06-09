from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np
from PIL import Image
import io
import requests
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import pydotplus
from io import StringIO
import base64
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime
import time
import random
import uuid
import shutil
import sqlite3
from PIL import Image, ImageDraw, ImageFont
from routes.db import login_required, init_user_db, get_user_db_path
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pickle
from sentence_transformers import SentenceTransformer
import openai

# Import dei blueprint e funzioni di autenticazione
from routes.chatbot import chatbot
from routes.chatbot2 import chatbot2
from routes.learning import learning
from routes.resources import resources
from routes.machine_learning import machine_learning_bp
from routes.mathematics import mathematics

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configurazione cartella upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Funzione per creare thumbnail delle immagini
def create_thumbnail(file_path, thumbnail_path, size=(100, 100)):
    """Crea una miniatura per le immagini"""
    try:
        img = Image.open(file_path)
        img.thumbnail(size)
        img.save(thumbnail_path)
        return True
    except Exception as e:
        print(f"Errore nella creazione della miniatura: {e}")
        return False

# Fallback image generation function
def generate_fallback_image(prompt, aspect_ratio='1:1'):
    try:
        # Convert aspect ratio to dimensions
        ratio_map = {
            '1:1': (1024, 1024),  # Square
            '3:2': (1216, 832),   # Landscape
            '2:3': (832, 1216),   # Portrait
            '16:9': (1536, 640)   # Widescreen
        }
        width, height = ratio_map.get(aspect_ratio, (1024, 1024))
        
        # Create a blank image with a gradient background
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Create a gradient background
        for y in range(height):
            r = int(200 + (y * 55 / height))
            g = int(200 + (y * 40 / height))
            b = int(220 + (y * 35 / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        # Add some random shapes for visual interest
        for _ in range(10):
            shape_type = random.choice(['circle', 'rectangle'])
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(50, 200)
            r = random.randint(100, 240)
            g = random.randint(100, 240)
            b = random.randint(100, 240)
            alpha = random.randint(30, 100)
            fill_color = (r, g, b, alpha)
            
            if shape_type == 'circle':
                draw.ellipse((x, y, x + size, y + size), fill=fill_color, outline=None)
            else:
                draw.rectangle((x, y, x + size, y + size), fill=fill_color, outline=None)
        
        # Try to use a system font, fallback to default if not available
        try:
            font = ImageFont.truetype('Arial', 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Add the prompt as text
        prompt_text = f"Prompt: {prompt}"
        # Wrap text
        words = prompt_text.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            w, h = draw.textsize(test_line, font=font) if hasattr(draw, 'textsize') else (0, 0)
            if w > width - 40:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text
        y_text = height // 2 - len(lines) * 30 // 2
        for line in lines:
            w, h = draw.textsize(line, font=font) if hasattr(draw, 'textsize') else (0, 0)
            draw.text(((width - w) // 2, y_text), line, font=font, fill=(0, 0, 0))
            y_text += 30
        
        # Add a watermark
        watermark = "Immagine di fallback - Servizio AI temporaneamente non disponibile"
        w, h = draw.textsize(watermark, font=font) if hasattr(draw, 'textsize') else (0, 0)
        draw.text(((width - w) // 2, height - 50), watermark, font=font, fill=(80, 80, 80))
        
        # Save the image
        img_path = os.path.join('static', 'generated', f'fallback_img_{int(time.time())}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        img.save(img_path)
        
        return jsonify({
            'image_url': '/' + img_path,
            'prompt': prompt,
            'fallback': True
        })
    except Exception as e:
        print(f"Error in fallback image generation: {str(e)}")
        return jsonify({'error': 'Impossibile generare l\'immagine di fallback'}), 500

from routes.chatbot2 import chatbot2
from routes.learning import learning
from routes.resources import resources
from routes.db import get_db, get_user_db, register_user, authenticate_user, login_required, update_api_credits, get_api_credits, get_user_db_path

# Configurazione dell'applicazione
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chiave_segreta_predefinita')
app.config['USER_DB_DIR'] = os.path.join(os.path.dirname(__file__), 'user_databases')
app.config['SMTP_SERVER'] = 'smtp.gmail.com'
app.config['SMTP_PORT'] = 587
app.config['SMTP_USERNAME'] = 'your-email@gmail.com'  # Sostituire con l'email reale
app.config['SMTP_PASSWORD'] = 'your-app-password'  # Sostituire con la password dell'app

# Assicurati che la directory per i database degli utenti esista
os.makedirs(app.config['USER_DB_DIR'], exist_ok=True)

load_dotenv()

# Funzione per aggiornare i crediti API nella sessione prima di ogni richiesta
@app.before_request
def update_session_api_credits():
    """Aggiorna i crediti API nella sessione prima di ogni richiesta"""
    if 'user_id' in session and 'username' in session:
        # Aggiorna i crediti API solo ogni 5 secondi per evitare troppe query al database
        last_update = session.get('api_credits_last_update', 0)
        current_time = time.time()
        
        print(f"DEBUG - Checking if session needs API credits update. Time since last update: {current_time - last_update} seconds")
        
        if current_time - last_update > 5:  # Aggiorna ogni 5 secondi
            username = session.get('username')
            print(f"DEBUG - Updating API credits in session for user: {username}")
            api_credits = get_api_credits(username)
            print(f"DEBUG - Retrieved API credits: {api_credits}")
            
            # Salva i crediti nella sessione
            session['api_credits'] = api_credits
            session['api_credits_last_update'] = current_time
            print(f"DEBUG - Session updated with API credits: {session.get('api_credits')}")

# Endpoint per aggiornare manualmente i crediti API
@app.route('/api/refresh-credits', methods=['POST'])
@login_required
def refresh_api_credits():
    """Endpoint per aggiornare manualmente i crediti API"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'error': 'Utente non autenticato'}), 401
            
        # Forza l'aggiornamento dei crediti API nella sessione
        api_credits = get_api_credits(username)
        session['api_credits'] = api_credits
        session['api_credits_last_update'] = time.time()
        
        return jsonify({
            'success': True, 
            'credits': api_credits,
            'message': 'Crediti API aggiornati con successo'
        })
    except Exception as e:
        print(f"Errore nell'aggiornamento manuale dei crediti API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Registrazione dei blueprint
app.register_blueprint(chatbot)
app.register_blueprint(chatbot2)
app.register_blueprint(learning)
app.register_blueprint(resources)
app.register_blueprint(machine_learning_bp)
app.register_blueprint(mathematics)

# Variabili globali per i modelli
model = None
le = None
image_model = None
class_names = []
X_train = None
feature_names = None
training_data = []
IMG_SIZE = 224  # Dimensione standard per le immagini

# Assicurati che la cartella uploads esista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/15")
        print(f"{'='*50}")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 5 == 0:  # Mostra il progresso ogni 5 batch
            print(f"Batch {batch}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/15 completata:")
        print(f"Training accuracy: {logs['accuracy']:.4f}")
        print(f"Training loss: {logs['loss']:.4f}")
        print(f"Validation accuracy: {logs['val_accuracy']:.4f}")
        print(f"Validation loss: {logs['val_loss']:.4f}")
        print(f"{'='*50}\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = None
    
    if 'user_id' in session and request.method == 'GET':
        return redirect(url_for('profile'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            error = 'Username e password sono obbligatori'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': error})
        else:
            success_auth, message = authenticate_user(username, password)
            if success_auth:
                # Ottieni l'ID dell'utente e altre informazioni dal database
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute("SELECT id, email, avatar FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()
                conn.close()
                
                # Inizializza il database utente
                if not init_user_db(username):
                    error = "Errore nell'inizializzazione del database utente"
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({'success': False, 'error': error})
                    return render_template('login.html', error=error)
                
                # Imposta le variabili di sessione
                session['user_id'] = user[0]
                session['username'] = username
                session['email'] = user[1]
                session['avatar'] = user[2]
                
                # Carica i crediti API nella sessione
                api_credits = get_api_credits(username)
                session['api_credits'] = api_credits
                
                success = 'Login effettuato con successo!'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'success': True,
                        'message': success,
                        'redirect': url_for('profile')
                    })
                return redirect(url_for('profile'))
            else:
                error = message
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': error})
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': False, 'error': error})
    return render_template('login.html', error=error, success=success)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    
    if 'user_id' in session:
        return redirect(url_for('profile'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password or not confirm_password:
            error = 'Tutti i campi sono obbligatori'
        elif password != confirm_password:
            error = 'Le password non corrispondono'
        else:
            success_reg, message = register_user(username, password)
            if success_reg:
                success = message
            else:
                error = message
    
    return render_template('register.html', error=error, success=success)

@app.route('/logout')
def logout():
    # Rimuovi tutte le variabili di sessione
    session.clear()
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    username = session.get('username')
    error = None
    success = None
    
    # Ottieni informazioni sull'utente
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    # Gestisci l'aggiornamento del profilo
    if request.method == 'POST':
        # Gestisci l'upload dell'avatar
        if 'avatar' in request.files:
            avatar_file = request.files['avatar']
            if avatar_file and avatar_file.filename != '':
                # Verifica l'estensione del file
                allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
                file_ext = avatar_file.filename.rsplit('.', 1)[1].lower() if '.' in avatar_file.filename else ''
                
                if file_ext in allowed_extensions:
                    # Crea la directory per gli avatar se non esiste
                    avatar_dir = os.path.join(app.static_folder, 'avatars')
                    os.makedirs(avatar_dir, exist_ok=True)
                    
                    # Genera un nome file sicuro e unico
                    filename = f"user_{session['user_id']}_{int(time.time())}.{file_ext}"
                    filepath = os.path.join(avatar_dir, filename)
                    
                    # Salva il file
                    avatar_file.save(filepath)
                    
                    # Aggiorna il database con il percorso dell'avatar
                    avatar_url = url_for('static', filename=f'avatars/{filename}')
                    cursor.execute("UPDATE users SET avatar = ? WHERE username = ?", (avatar_url, username))
                    conn.commit()
                    
                    # Aggiorna la sessione
                    session['user_avatar'] = avatar_url
                    
                    success = "Avatar aggiornato con successo!"
                else:
                    error = "Formato file non supportato. Usa PNG, JPG, JPEG o GIF."
        
        # Aggiorna le informazioni del profilo
        firstname = request.form.get('firstname', '')
        lastname = request.form.get('lastname', '')
        email = request.form.get('email', '')
        
        # Aggiorna il database
        cursor.execute("UPDATE users SET firstname = ?, lastname = ?, email = ? WHERE username = ?", 
                      (firstname, lastname, email, username))
        conn.commit()
        
        # Aggiorna la sessione
        session['user_email'] = email
        
        if not error:
            success = "Profilo aggiornato con successo!"
    
    # Ottieni le informazioni aggiornate dell'utente
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    # Ottieni i crediti API utilizzati dall'utente
    api_credits = get_api_credits(username)
    
    conn.close()
    
    # Ottieni i dati dell'utente dal suo database personalizzato
    user_conn = get_user_db(username)
    user_cursor = user_conn.cursor()
    
    # Verifica se la tabella user_data esiste e creala se necessario
    user_cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_type TEXT,
        data_name TEXT,
        data_value TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    user_conn.commit()
    
    # Ora possiamo eseguire la query in sicurezza
    user_cursor.execute("SELECT * FROM user_data ORDER BY created_at DESC")
    user_data = user_cursor.fetchall()
    user_conn.close()
    
    # Formatta i dati utente per renderli accessibili nel template
    user_dict = {}
    if user:
        for key in user.keys():
            user_dict[key] = user[key]
    
    # Passa eventuali messaggi di errore o successo dalla query string
    if not error:
        error = request.args.get('error')
    if not success:
        success = request.args.get('success')
    
    return render_template('profile.html', 
                           username=username, 
                           user=user_dict,
                           created_at=user['created_at'], 
                           user_data=user_data,
                           error=error,
                           success=success,
                           api_credits=api_credits)

@app.route('/save_user_data', methods=['POST'])
@login_required
def save_user_data():
    username = session.get('username')
    data_type = request.form.get('data_type')
    data_name = request.form.get('data_name')
    data_value = request.form.get('data_value')
    
    if not data_type or not data_name or not data_value:
        return redirect(url_for('profile', error='Tutti i campi sono obbligatori'))
    
    try:
        conn = get_user_db(username)
        cursor = conn.cursor()
        
        # Verifica se la tabella user_data esiste e creala se necessario
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type TEXT,
            data_name TEXT,
            data_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        
        cursor.execute(
            "INSERT INTO user_data (data_type, data_name, data_value) VALUES (?, ?, ?)",
            (data_type, data_name, data_value)
        )
        conn.commit()
        conn.close()
        
        return redirect(url_for('profile', success='Dati salvati con successo'))
    except Exception as e:
        return redirect(url_for('profile', error=f'Errore durante il salvataggio: {str(e)}'))

@app.route('/view_user_data/<int:data_id>')
@login_required
def view_user_data(data_id):
    username = session.get('username')
    
    try:
        conn = get_user_db(username)
        cursor = conn.cursor()
        
        # Verifica se la tabella user_data esiste e creala se necessario
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type TEXT,
            data_name TEXT,
            data_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        
        cursor.execute("SELECT * FROM user_data WHERE id = ?", (data_id,))
        data = cursor.fetchone()
        conn.close()
        
        if not data:
            return redirect(url_for('profile', error='Dati non trovati'))
        
        return render_template('view_data.html', data=data)
    except Exception as e:
        return redirect(url_for('profile', error=f'Errore durante la visualizzazione dei dati: {str(e)}'))

@app.route('/delete_user_data/<int:data_id>')
@login_required
def delete_user_data(data_id):
    username = session.get('username')
    
    try:
        conn = get_user_db(username)
        cursor = conn.cursor()
        
        # Verifica se la tabella user_data esiste e creala se necessario
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type TEXT,
            data_name TEXT,
            data_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        
        cursor.execute("DELETE FROM user_data WHERE id = ?", (data_id,))
        conn.commit()
        conn.close()
        
        return redirect(url_for('profile', success='Dati eliminati con successo'))
    except Exception as e:
        return redirect(url_for('profile', error=f'Errore durante l\'eliminazione: {str(e)}'))

@app.route('/regressione')
def regressione():
    return render_template('regressione.html')

@app.route('/classificazione')
def classificazione():
    return render_template('classificazione.html')

@app.route('/classificazione-immagini')
def classificazione_immagini():
    return render_template('classificazione_immagini.html')

@app.route('/generazione')
def generazione():
    return render_template('generazione_immagini.html')

@app.route('/class_img_2')
def class_img_2():
    return render_template('class_img_2.html')

@app.route('/visione')
def visione():
    return render_template('class_img_2.html')

@app.route('/upload-regression', methods=['POST'])
def upload_regression():
    print("DEBUG: Inizio upload_regression")
    if 'file' not in request.files and 'file_content' not in request.form:
        print("DEBUG: Nessun file caricato")
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    # Ottieni il tipo di modello richiesto (default: linear)
    model_type = request.form.get('model_type', 'linear')
    print(f"DEBUG: Tipo di modello: {model_type}")
    
    try:
        # Carica i dati dal file o dal contenuto
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            print(f"DEBUG: Caricamento file: {file.filename}")
            try:
                # Leggi il contenuto del file come stringa
                file_content = file.read().decode('utf-8')
                print(f"DEBUG: Contenuto file (primi 100 caratteri): {file_content[:100]}")
                # Usa StringIO per leggere il CSV con opzioni specifiche per gestire i numeri decimali
                df = pd.read_csv(
                    StringIO(file_content),
                    decimal='.',  # Usa il punto come separatore decimale
                    thousands=None,  # Non usare separatori per le migliaia
                    float_precision='high'  # Usa precisione alta per i numeri
                )
            except Exception as e:
                print(f"DEBUG: Errore nella lettura del CSV: {str(e)}")
                return jsonify({'error': f'Errore nella lettura del file CSV: {str(e)}'}), 400
        elif 'file_content' in request.form:
            content = request.form['file_content']
            print(f"DEBUG: Contenuto form (primi 100 caratteri): {content[:100]}")
            # Usa le stesse opzioni di parsing per i dati forniti tramite form
            df = pd.read_csv(
                StringIO(content),
                decimal='.',  # Usa il punto come separatore decimale
                thousands=None,  # Non usare separatori per le migliaia
                float_precision='high'  # Usa precisione alta per i numeri
            )
        else:
            print("DEBUG: Nessun file selezionato")
            return jsonify({'error': 'Nessun file selezionato'}), 400
            
        print(f"DEBUG: Colonne del DataFrame: {df.columns.tolist()}")
        print(f"DEBUG: Tipi di dati: {df.dtypes}")
        print(f"DEBUG: Prime 5 righe:\n{df.head()}")
        
        if len(df.columns) != 2:
            print(f"DEBUG: Numero di colonne errato: {len(df.columns)}")
            return jsonify({'error': 'Il file deve contenere esattamente due colonne'}), 400

        try:
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 1].values
            
            # Converti i dati di training in una lista di coppie [x, y]
            training_data = []
            for x, y in zip(X.flatten(), y):
                try:
                    x_float = float(x)
                    y_float = float(y)
                    training_data.append([x_float, y_float])
                except Exception as e:
                    print(f"DEBUG: Errore nella conversione dei valori: {x}, {y}, {str(e)}")
                    return jsonify({'error': f'Errore nella conversione dei valori: {str(e)}'}), 400
            
            print(f"DEBUG: Numero di punti dati: {len(training_data)}")
        except Exception as e:
            print(f"DEBUG: Errore nell'elaborazione dei dati: {str(e)}")
            return jsonify({'error': f'Errore nell\'elaborazione dei dati: {str(e)}'}), 400
        
        # Crea il modello in base al tipo richiesto
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            # Per la regressione lineare, restituisci coefficiente e intercetta
            return jsonify({
                'success': True,
                'model_type': 'linear',
                'columns': df.columns.tolist(),
                'coefficiente': float(model.coef_[0]),
                'intercetta': float(model.intercept_),
                'training_data': training_data
            })
        
        elif model_type == 'polynomial':
            # Ottieni il grado del polinomio (default: 2)
            degree = int(request.form.get('degree', 2))
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X, y)
            
            # Genera punti per la curva polinomiale
            x_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            curve_points = [[float(x), float(y)] for x, y in zip(x_range.flatten(), y_pred)]
            
            return jsonify({
                'success': True,
                'model_type': 'polynomial',
                'degree': degree,
                'columns': df.columns.tolist(),
                'training_data': training_data,
                'curve_points': curve_points,
                'model_params': {
                    'degree': degree
                }
            })
        
        elif model_type == 'svr':
            # Parametri SVR
            kernel = request.form.get('kernel', 'rbf')
            C = float(request.form.get('C', 1.0))
            epsilon = float(request.form.get('epsilon', 0.1))
            
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            model.fit(X, y)
            
            # Genera punti per la curva SVR
            x_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            curve_points = [[float(x), float(y)] for x, y in zip(x_range.flatten(), y_pred)]
            
            return jsonify({
                'success': True,
                'model_type': 'svr',
                'columns': df.columns.tolist(),
                'training_data': training_data,
                'curve_points': curve_points,
                'model_params': {
                    'kernel': kernel,
                    'C': C,
                    'epsilon': epsilon
                }
            })
            
        elif model_type == 'random_forest':
            # Parametri Random Forest
            n_estimators = int(request.form.get('n_estimators', 100))
            max_depth = request.form.get('max_depth', 'None')
            if max_depth != 'None':
                max_depth = int(max_depth)
            else:
                max_depth = None
                
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X, y)
            
            # Genera punti per la curva Random Forest
            x_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            curve_points = [[float(x), float(y)] for x, y in zip(x_range.flatten(), y_pred)]
            
            return jsonify({
                'success': True,
                'model_type': 'random_forest',
                'columns': df.columns.tolist(),
                'training_data': training_data,
                'curve_points': curve_points,
                'model_params': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
            })
            
        elif model_type == 'decision_tree':
            # Parametri Decision Tree
            max_depth = request.form.get('max_depth', 'None')
            if max_depth != 'None':
                max_depth = int(max_depth)
            else:
                max_depth = None
                
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            model.fit(X, y)
            
            # Genera punti per la curva Decision Tree
            x_range = np.linspace(min(X.flatten()), max(X.flatten()), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            curve_points = [[float(x), float(y)] for x, y in zip(x_range.flatten(), y_pred)]
            
            return jsonify({
                'success': True,
                'model_type': 'decision_tree',
                'columns': df.columns.tolist(),
                'training_data': training_data,
                'curve_points': curve_points,
                'model_params': {
                    'max_depth': max_depth
                }
            })
        
        else:
            return jsonify({'error': f'Tipo di modello non supportato: {model_type}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-regression', methods=['POST'])
def predict_regression():
    try:
        data = request.json
        value = float(data['value'])
        model_type = data.get('model_type', 'linear')
        
        if model_type == 'linear':
            # Per regressione lineare, usa coefficiente e intercetta
            coef = float(data['coefficiente'])
            intercept = float(data['intercetta'])
            prediction = coef * value + intercept
            return jsonify({'prediction': prediction})
            
        elif model_type in ['polynomial', 'svr', 'random_forest', 'decision_tree']:
            # Per modelli non lineari, usa i punti della curva per interpolare
            curve_points = data['curve_points']
            x_values = [point[0] for point in curve_points]
            y_values = [point[1] for point in curve_points]
            
            # Trova il punto più vicino o interpola
            if value <= min(x_values):
                prediction = y_values[0]
            elif value >= max(x_values):
                prediction = y_values[-1]
            else:
                # Interpolazione lineare tra i punti più vicini
                for i in range(len(x_values) - 1):
                    if x_values[i] <= value <= x_values[i + 1]:
                        # Calcola l'interpolazione lineare
                        t = (value - x_values[i]) / (x_values[i + 1] - x_values[i])
                        prediction = y_values[i] * (1 - t) + y_values[i + 1] * t
                        break
            
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': f'Tipo di modello non supportato: {model_type}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload-classification', methods=['POST'])
def upload_classification():
    global model, le, X_train, feature_names, training_data, tree_data
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400

    try:
        df = pd.read_csv(file)
        
        # Validate dataset
        if len(df.columns) < 2:
            return jsonify({'error': 'Il dataset deve avere almeno due colonne (features e target)'}), 400

        # Ensure target column is string type
        df.iloc[:, -1] = df.iloc[:, -1].astype(str)

        # Check for sufficient data
        if len(df) < 5:
            return jsonify({'error': 'Insufficienti dati per il training (minimo 5 esempi richiesti)'}), 400

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Salva feature names per future visualizzazioni
        feature_names = X.columns.tolist()
        X_train = X.values

        # Addestra il modello RandomForest
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y_encoded)
        
        # Crea e addestra un albero decisionale per la visualizzazione
        tree_model = DecisionTreeClassifier(max_depth=4)
        tree_model.fit(X, y_encoded)
        
        # Genera la visualizzazione dell'albero in formato DOT
        dot_data = StringIO()
        export_graphviz(tree_model, out_file=dot_data, 
                        feature_names=feature_names,
                        class_names=le.classes_.tolist(),
                        filled=True, rounded=True,
                        special_characters=True,
                        impurity=False)
        
        # Ottieni la rappresentazione grafica
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        tree_png = graph.create_png()
        
        # Codifica l'immagine in base64 per il frontend
        tree_data = base64.b64encode(tree_png).decode('utf-8')

        # Prepara i dati di addestramento per la visualizzazione
        training_data = []
        for i in range(len(X)):
            training_data.append({
                'features': X.iloc[i].tolist(),
                'target': y.iloc[i]
            })

        return jsonify({
            'success': True,
            'columns': feature_names,
            'classes': le.classes_.tolist(),
            'training_data': training_data,
            'tree_image': tree_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-classification', methods=['POST'])
def predict_classification():
    global model, le, X_train, feature_names
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Nessun dato ricevuto'}), 400

        # Estrai i nomi delle caratteristiche e i valori di input
        input_features = list(data.keys())
        input_values = [float(value) for value in data.values()]
        
        # Converti i dati in un formato adatto per la predizione
        features = np.array([input_values])
        
        # Usa il modello per fare la predizione
        prediction = model.predict(features)[0]
        predicted_class = le.inverse_transform([prediction])[0]
        
        # Prepara la risposta con i dati per il grafico
        response = {
            'prediction': str(predicted_class),
            'input_values': input_values,
            'features': input_features
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train-image-classifier', methods=['POST'])
def train_image_classifier():
    global image_model, class_names
    try:
        if 'images[]' not in request.files and 'webcam_images[]' not in request.files:
            return jsonify({'error': 'Nessuna immagine caricata'}), 400

        # Gestione immagini da file e webcam
        images = request.files.getlist('images[]') if 'images[]' in request.files else []
        webcam_images = request.files.getlist('webcam_images[]') if 'webcam_images[]' in request.files else []
        all_images = images + webcam_images
        
        classes = request.form.getlist('classes[]')
        
        if len(all_images) == 0 or len(classes) == 0:
            return jsonify({'error': 'Dati mancanti'}), 400
        
        # Crea cartelle temporanee per le classi
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_training')
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Organizza le immagini nelle cartelle per classe
        class_names = sorted(list(set(classes)))  # Ordina le classi alfabeticamente
        print(f"Classi trovate: {class_names}")  # Debug
        
        for class_name in class_names:
            os.makedirs(os.path.join(temp_dir, class_name), exist_ok=True)

        # Salva le immagini nelle rispettive cartelle
        for img, class_name in zip(all_images, classes):
            if img.filename:
                # Preprocessa l'immagine
                image = Image.open(img)
                image = image.convert('RGB')
                image = image.resize((IMG_SIZE, IMG_SIZE))
                
                # Salva l'immagine preprocessata
                img_path = os.path.join(temp_dir, class_name, secure_filename(img.filename))
                image.save(img_path)
                print(f"Salvata immagine {img.filename} per la classe {class_name}")  # Debug

        # Configurazione del data generator
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Preparazione dei dataset
        train_generator = datagen.flow_from_directory(
            temp_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            classes=class_names  # Specifica l'ordine delle classi
        )

        validation_generator = datagen.flow_from_directory(
            temp_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            classes=class_names  # Specifica l'ordine delle classi
        )

        print(f"Mappatura classi: {train_generator.class_indices}")  # Debug

        # Creazione del modello
        image_model = Sequential([
            Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])

        # Compilazione del modello
        image_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Training del modello con il custom callback
        custom_callback = CustomCallback()
        history = image_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=15,
            verbose=1,
            callbacks=[custom_callback]
        )

        # Salva il modello e i nomi delle classi
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image_model.h5')
        image_model.save(model_path)
        
        # Salva anche i nomi delle classi
        import json
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'class_names.json'), 'w') as f:
            json.dump(class_names, f)

        return jsonify({
            'success': True,
            'training_accuracy': float(history.history['accuracy'][-1]),
            'validation_accuracy': float(history.history['val_accuracy'][-1]),
            'classes': class_names
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug
        return jsonify({'error': str(e)}), 400

@app.route('/predict-image', methods=['POST'])
def predict_image():
    global image_model, class_names
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nessuna immagine caricata'}), 400

        if image_model is None:
            return jsonify({'error': 'Modello non ancora addestrato'}), 400

        file = request.files['image']
        
        # Preprocessa l'immagine
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Converti l'immagine in array
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Fai la predizione
        predictions = image_model.predict(img_array)
        
        # Crea una lista di tutte le classi con le loro confidenze
        results = []
        for idx, confidence in enumerate(predictions[0]):
            results.append({
                'class': class_names[idx],
                'confidence': float(confidence) * 100  # Converti in percentuale
            })
        
        # Ordina i risultati per confidenza decrescente
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'predictions': results,
            'top_prediction': results[0]['class']
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt')
        style = data.get('style', 'photographic')
        aspect_ratio = data.get('aspect_ratio', '1:1')
        high_quality = data.get('high_quality', False)
        model_choice = data.get('model', 'stability')
        
        # Choose between Stability AI and DALL-E models
        if model_choice == 'dalle':
            return generate_dalle_image(prompt, style, aspect_ratio, high_quality)
        else:
            # Default to Stability AI
            model = 'stable-diffusion-xl-1024-v1-0'
        
        # Convert aspect ratio to dimensions
        ratio_map = {
            '1:1': (1024, 1024),  # Square
            '3:2': (1216, 832),   # Landscape
            '2:3': (832, 1216),   # Portrait
            '16:9': (1536, 640)   # Widescreen
        }
        width, height = ratio_map.get(aspect_ratio, (1024, 1024))

        # Add style to prompt if specified
        if style != 'photographic':
            style_prompts = {
                'digital-art': 'digital art style',
                'oil-painting': 'oil painting style',
                'watercolor': 'watercolor painting style',
                'anime': 'anime style',
                '3d-render': '3D rendered style'
            }
            prompt += f", {style_prompts.get(style, '')}"

        # Prepare the request to Stability AI API
        url = f"https://api.stability.ai/v1/generation/{model}/text-to-image"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"
        }
        
        # Set parameters for the most performant model
        steps = 40 if high_quality else 30
        cfg_scale = 7
        
        body = {
            "width": width,
            "height": height,
            "steps": steps,
            "seed": 0,
            "cfg_scale": cfg_scale,
            "samples": 1,
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1
                }
            ],
        }

        # Make the API request
        response = requests.post(url, headers=headers, json=body)
        
        # Aggiorna i crediti API se l'utente è loggato
        if session.get('user_id') and session.get('username'):
            update_api_credits(session.get('username'), 'stability')
        
        if response.status_code != 200:
            try:
                if response.headers.get('content-type') == 'application/json':
                    error_data = response.json()
                    error_message = error_data.get('message', 'Errore sconosciuto')
                else:
                    # Check if it's a Cloudflare error page
                    if 'cloudflare' in response.text.lower() or 'error code 520' in response.text.lower():
                        error_message = 'Errore di connessione al server Stability AI. Servizio temporaneamente non disponibile.'
                    else:
                        error_message = 'Errore sconosciuto'
                
                if 'API key' in error_message:
                    error_message = 'Errore di autenticazione: chiave API non valida o mancante'
                elif 'insufficient balance' in error_message.lower():
                    error_message = 'Credito insufficiente per generare l\'immagine'
                elif 'invalid parameter' in error_message.lower():
                    error_message = 'Parametri di generazione non validi'
            except Exception:
                error_message = 'Errore di connessione al server Stability AI. Servizio temporaneamente non disponibile.'
                
                # Use fallback image generation
                return generate_fallback_image(prompt, aspect_ratio)

        # Process the response
        try:
            data = response.json()
            
            if "artifacts" not in data or len(data["artifacts"]) == 0:
                return generate_fallback_image(prompt, aspect_ratio)
        except Exception:
            # If we can't parse the response as JSON, use the fallback
            return generate_fallback_image(prompt, aspect_ratio)
            
        # Save the first generated image
        image_data = data["artifacts"][0]
        image_bytes = base64.b64decode(image_data["base64"])
        
        # Save the image
        img_path = os.path.join('static', 'generated', f'img_{int(time.time())}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        with open(img_path, 'wb') as f:
            f.write(image_bytes)
        
        return jsonify({
            'image_url': '/' + img_path,
            'prompt': prompt
        })

    except ValueError as e:
        print(f"Validation error in generate_image: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error in generate_image: {str(e)}")
        return jsonify({'error': 'Si è verificato un errore imprevisto durante la generazione dell\'immagine'}), 500


def generate_dalle_image(prompt, style, aspect_ratio, high_quality):
    try:
        # Convert aspect ratio to dimensions
        ratio_map = {
            '1:1': '1024x1024',  # Square
            '3:2': '1792x1024',  # Landscape
            '2:3': '1024x1792',  # Portrait
            '16:9': '1792x1024'  # Widescreen (using closest available)
        }
        size = ratio_map.get(aspect_ratio, '1024x1024')
        
        # Add style to prompt if specified
        if style != 'photographic':
            style_prompts = {
                'digital-art': 'digital art style',
                'oil-painting': 'oil painting style',
                'watercolor': 'watercolor painting style',
                'anime': 'anime style',
                '3d-render': '3D rendered style'
            }
            prompt += f", {style_prompts.get(style, '')}"
        
        # Choose model quality based on high_quality flag
        model = "dall-e-3" if high_quality else "dall-e-2"
        
        # Prepare the request to OpenAI API
        url = "https://api.openai.com/v1/images/generations"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        body = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json"
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=body)
        
        # Aggiorna i crediti API se l'utente è loggato
        if session.get('user_id') and session.get('username'):
            update_api_credits(session.get('username'), 'stability')
        
        if response.status_code != 200:
            try:
                if response.headers.get('content-type') == 'application/json':
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'Errore sconosciuto')
                else:
                    error_message = 'Errore sconosciuto'
                
                if 'API key' in error_message:
                    error_message = 'Errore di autenticazione: chiave API OpenAI non valida o mancante'
                elif 'insufficient balance' in error_message.lower() or 'quota' in error_message.lower():
                    error_message = 'Credito insufficiente per generare l\'immagine con OpenAI'
                elif 'content policy' in error_message.lower():
                    error_message = 'Il prompt viola le politiche sui contenuti di OpenAI'
            except Exception:
                # Use fallback image generation on any exception
                return generate_fallback_image(prompt, aspect_ratio)
            
            # Use fallback image generation for all API errors
            return generate_fallback_image(prompt, aspect_ratio)
        
        # Process the response
        try:
            data = response.json()
            
            if "data" not in data or len(data["data"]) == 0:
                return generate_fallback_image(prompt, aspect_ratio)
                
            # Get the image data
            image_data = data["data"][0]
            
            if "b64_json" in image_data:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data["b64_json"])
                
                # Save the image
                img_path = os.path.join('static', 'generated', f'dalle_img_{int(time.time())}.png')
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                with open(img_path, 'wb') as f:
                    f.write(image_bytes)
                
                return jsonify({
                    'image_url': '/' + img_path,
                    'prompt': prompt,
                    'model': 'dalle'
                })
            elif "url" in image_data:
                # If we got a URL instead of base64 data, download the image
                img_url = image_data["url"]
                img_response = requests.get(img_url)
                
                if img_response.status_code == 200:
                    # Save the image
                    img_path = os.path.join('static', 'generated', f'dalle_img_{int(time.time())}.png')
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    return jsonify({
                        'image_url': '/' + img_path,
                        'prompt': prompt,
                        'model': 'dalle'
                    })
        except Exception:
            # If we can't parse the response as JSON, use the fallback
            return generate_fallback_image(prompt, aspect_ratio)
        
        # If we get here, something went wrong
        return generate_fallback_image(prompt, aspect_ratio)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/save-image-to-resources', methods=['POST'])
@login_required
def save_image_to_resources():
    try:
        # Get data from request
        data = request.json
        image_url = data.get('image_url')
        prompt = data.get('prompt')
        model = data.get('model', 'stability')
        style = data.get('style', 'photographic')
        
        if not image_url:
            return jsonify({'error': 'URL dell\'immagine non fornito'}), 400
        
        # Extract the path from the URL
        if image_url.startswith('/'):
            image_path = image_url[1:]  # Remove leading slash
        else:
            image_path = image_url
            
        # For debugging
        print(f"Image URL: {image_url}")
        print(f"Image path: {image_path}")
        
        # If the image path doesn't exist, try to find it in the static/generated directory
        if not os.path.exists(image_path) and '/static/generated/' in image_url:
            # Extract the filename from the URL
            filename = os.path.basename(image_url)
            alternative_path = os.path.join('static', 'generated', filename)
            print(f"Trying alternative path: {alternative_path}")
            
            if os.path.exists(alternative_path):
                image_path = alternative_path
                print(f"Found image at alternative path: {image_path}")
        
        # Check if the file exists
        if not os.path.exists(image_path):
            # Try to get the image from the URL directly if it's a data URL
            if image_url.startswith('data:image/'):
                try:
                    # Parse the data URL
                    header, encoded = image_url.split(',', 1)
                    data = base64.b64decode(encoded)
                    
                    # Generate a unique filename for the resource
                    file_id = str(uuid.uuid4())
                    extension = '.png'  # Generated images are always PNG
                    
                    # Create the destination path in the resources folder
                    dest_path = os.path.join('static', 'uploads', 'resources', f"{file_id}{extension}")
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Save the image data directly
                    with open(dest_path, 'wb') as f:
                        f.write(data)
                    
                    # Create a thumbnail
                    thumbnail_path = os.path.join('static', 'uploads', 'resources', f"{file_id}_thumb{extension}")
                    create_thumbnail(dest_path, thumbnail_path)
                    
                    # Get file size
                    file_size = os.path.getsize(dest_path)
                    
                    # Generate a descriptive name
                    model_name = "DALL-E" if model == "dalle" else "Stability AI"
                    filename = f"Immagine generata con {model_name} - {style} - {datetime.now().strftime('%Y-%m-%d %H:%M')}.png"
                    
                    # Save to database
                    conn = sqlite3.connect('database.sqlite')
                    c = conn.cursor()
                    c.execute('''INSERT INTO resources (id, user_id, name, original_name, path, type, size, date)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                             (file_id, session['user_id'], f"{file_id}{extension}", filename, dest_path, 'image/png', file_size, datetime.now()))
                    conn.commit()
                    conn.close()
                    
                    return jsonify({
                        'success': True,
                        'message': 'Immagine salvata con successo nelle risorse',
                        'resource_id': file_id
                    })
                except Exception as e:
                    print(f"Errore nel salvataggio dell'immagine data URL: {str(e)}")
                    return jsonify({'error': f'Errore nel salvataggio dell\'immagine: {str(e)}'}), 500
            else:
                return jsonify({'error': 'Immagine non trovata'}), 404
            
        # We already handled the data URL case above
        # This code will only execute if the image file exists on the server
        
        # Generate a unique filename for the resource
        file_id = str(uuid.uuid4())
        extension = '.png'  # Generated images are always PNG
        
        # Create the destination path in the resources folder
        dest_path = os.path.join('static', 'uploads', 'resources', f"{file_id}{extension}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(image_path, dest_path)
        
        # Create a thumbnail
        thumbnail_path = os.path.join('static', 'uploads', 'resources', f"{file_id}_thumb{extension}")
        create_thumbnail(dest_path, thumbnail_path)
        
        # Get file size
        file_size = os.path.getsize(dest_path)
        
        # Generate a descriptive name
        model_name = "DALL-E" if model == "dalle" else "Stability AI"
        filename = f"Immagine generata con {model_name} - {style} - {datetime.now().strftime('%Y-%m-%d %H:%M')}.png"
        
        # Save to database
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        c.execute('''INSERT INTO resources (id, user_id, name, original_name, path, type, size, date)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                 (file_id, session['user_id'], f"{file_id}{extension}", filename, dest_path, 'image/png', file_size, datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Immagine salvata con successo nelle risorse',
            'resource_id': file_id
        })
        
    except Exception as e:
        print(f"Errore nel salvataggio dell'immagine nelle risorse: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-image-from-image', methods=['POST'])
def generate_image_from_image():
    try:
        # Get form data
        prompt = request.form.get('prompt')
        style = request.form.get('style', 'photographic')
        aspect_ratio = request.form.get('aspect_ratio', '1:1')
        high_quality = request.form.get('high_quality') == 'true'
        
        # Check if an image was uploaded
        if 'image' not in request.files:
            raise ValueError('Nessuna immagine caricata')
            
        image_file = request.files['image']
        if image_file.filename == '':
            raise ValueError('Nessuna immagine selezionata')
            
        # Check file type
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError('Formato file non supportato. Usa PNG o JPG')
            
        # Read and encode the image
        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Use only the most performant model
        model = 'stable-diffusion-xl-1024-v1-0'
        
        # Convert aspect ratio to dimensions
        ratio_map = {
            '1:1': (1024, 1024),  # Square
            '3:2': (1216, 832),   # Landscape
            '2:3': (832, 1216),   # Portrait
            '16:9': (1536, 640)   # Widescreen
        }
        width, height = ratio_map.get(aspect_ratio, (1024, 1024))
        
        # Add style to prompt if specified
        if style != 'photographic':
            style_prompts = {
                'digital-art': 'digital art style',
                'oil-painting': 'oil painting style',
                'watercolor': 'watercolor painting style',
                'anime': 'anime style',
                '3d-render': '3D rendered style'
            }
            prompt += f", {style_prompts.get(style, '')}"
        
        # Prepare the request to Stability AI API for image-to-image
        url = f"https://api.stability.ai/v1/generation/{model}/image-to-image"
        
        # Set parameters for the most performant model
        steps = 40 if high_quality else 30
        cfg_scale = 7
        image_strength = 0.35  # How much influence the input image has (0.35 is a good balance)
        
        # Prepare multipart form data for the API request
        form_data = {
            'init_image': encoded_image,
            'text_prompts[0][text]': prompt,
            'text_prompts[0][weight]': '1',
            'image_strength': str(image_strength),
            'cfg_scale': str(cfg_scale),
            'samples': '1',
            'steps': str(steps),
            'seed': '0',
        }
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}"
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, data=form_data)
        
        if response.status_code != 200:
            try:
                if response.headers.get('content-type') == 'application/json':
                    error_data = response.json()
                    error_message = error_data.get('message', 'Errore sconosciuto')
                else:
                    # Check if it's a Cloudflare error page
                    if 'cloudflare' in response.text.lower() or 'error code 520' in response.text.lower():
                        error_message = 'Errore di connessione al server Stability AI. Servizio temporaneamente non disponibile.'
                    else:
                        error_message = 'Errore sconosciuto'
                
                if 'API key' in error_message:
                    error_message = 'Errore di autenticazione: chiave API non valida o mancante'
                elif 'insufficient balance' in error_message.lower():
                    error_message = 'Credito insufficiente per generare l\'immagine'
                elif 'invalid parameter' in error_message.lower():
                    error_message = 'Parametri di generazione non validi'
            except Exception:
                error_message = 'Errore di connessione al server Stability AI. Servizio temporaneamente non disponibile.'
                
                # Use fallback image generation
                return generate_fallback_image(prompt, aspect_ratio)
        
        # Process the response
        try:
            data = response.json()
            
            if "artifacts" not in data or len(data["artifacts"]) == 0:
                return generate_fallback_image(prompt, aspect_ratio)
        except Exception:
            # If we can't parse the response as JSON, use the fallback
            return generate_fallback_image(prompt, aspect_ratio)
            
        # Save the first generated image
        image_data = data["artifacts"][0]
        result_image_bytes = base64.b64decode(image_data["base64"])
        
        # Save the image
        img_path = os.path.join('static', 'generated', f'img_from_img_{int(time.time())}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        with open(img_path, 'wb') as f:
            f.write(result_image_bytes)
        
        return jsonify({
            'image_url': '/' + img_path,
            'prompt': prompt
        })
        
    except ValueError as e:
        print(f"Validation error in generate_image_from_image: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error in generate_image_from_image: {str(e)}")
        return jsonify({'error': 'Si è verificato un errore imprevisto durante la generazione dell\'immagine'}), 500

@app.route('/api/translate-enhance-prompt', methods=['POST'])
def translate_enhance_prompt():
    try:
        data = request.json
        prompt = data.get('prompt', '')

        # Prepare the message for GPT-4
        system_message = """Sei un esperto di prompt engineering per la generazione di immagini. 
        Il tuo compito è:
        1. Tradurre il prompt dall'italiano all'inglese se necessario
        2. Migliorare il prompt aggiungendo dettagli che possono aiutare a generare un'immagine migliore
        3. Aggiungere parametri tecnici come l'illuminazione, la composizione, la prospettiva, ecc.
        4. Mantenere uno stile naturale e fluido
        
        Rispondi SOLO con il prompt migliorato, senza spiegazioni o altro testo."""

        # Call GPT-4 for translation and enhancement
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        enhanced_prompt = response.choices[0].message.content.strip()

        return jsonify({
            'enhanced_prompt': enhanced_prompt
        })

    except Exception as e:
        print(f"Error in translate_enhance_prompt: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-image-classifier-v2', methods=['POST'])
def train_image_classifier_v2():
    try:
        data = request.json
        if not data or 'images' not in data or 'labels' not in data or 'classNames' not in data:
            return jsonify({'error': 'Dati mancanti'}), 400
        
        images = data['images']
        labels = data['labels']
        class_names = data['classNames']
        
        if len(images) == 0 or len(labels) == 0 or len(class_names) == 0:
            return jsonify({'error': 'Dati insufficienti per il training'}), 400
        
        # Genera un ID univoco per il modello
        model_id = f"model_{int(time.time())}"
        model_dir = os.path.join(app.config['UPLOAD_FOLDER'], model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Salva i nomi delle classi
        with open(os.path.join(model_dir, 'class_names.json'), 'w') as f:
            json.dump(class_names, f)
        
        # Prepara i dati di training
        X_train = []
        y_train = []
        
        for i, img_data in enumerate(images):
            try:
                # Rimuovi il prefisso 'data:image/jpeg;base64,' se presente
                if 'base64,' in img_data:
                    img_data = img_data.split('base64,')[1]
                
                # Decodifica l'immagine base64
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                img = img.convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                
                # Converti l'immagine in array e normalizza
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                
                X_train.append(img_array)
                y_train.append(labels[i])
            except Exception as e:
                print(f"Errore nel processare l'immagine {i}: {str(e)}")
        
        # Converti le liste in array numpy
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Dividi i dati in training e validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Crea il modello
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])

        # Compila il modello
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callback personalizzato per il logging
        callback = CustomCallback()
        
        # Addestra il modello
        history = model.fit(
            X_train, y_train,
            epochs=15,
            validation_data=(X_val, y_val),
            callbacks=[callback]
        )

        # Salva il modello
        model.save(os.path.join(model_dir, 'model.h5'))
        
        return jsonify({
            'success': True,
            'modelId': model_id,
            'training_accuracy': float(history.history['accuracy'][-1]),
            'validation_accuracy': float(history.history['val_accuracy'][-1]),
            'classes': class_names
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict-image-v2', methods=['POST'])
def predict_image_v2():
    try:
        data = request.json
        if not data or 'image' not in data or 'modelId' not in data:
            return jsonify({'error': 'Dati mancanti'}), 400
        
        model_id = data['modelId']
        model_dir = os.path.join(app.config['UPLOAD_FOLDER'], model_id)
        
        # Verifica che il modello esista
        if not os.path.exists(model_dir):
            return jsonify({'error': 'Modello non trovato'}), 404
        
        # Carica il modello
        model_path = os.path.join(model_dir, 'model.h5')
        if not os.path.exists(model_path):
            return jsonify({'error': 'File del modello non trovato'}), 404
        
        model = load_model(model_path)
        
        # Carica i nomi delle classi
        class_names_path = os.path.join(model_dir, 'class_names.json')
        if not os.path.exists(class_names_path):
            return jsonify({'error': 'File dei nomi delle classi non trovato'}), 404
        
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        # Processa l'immagine
        img_data = data['image']
        if 'base64,' in img_data:
            img_data = img_data.split('base64,')[1]
        
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Converti l'immagine in array e normalizza
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Fai la predizione
        predictions = model.predict(img_array)
        
        # Crea una lista di tutte le classi con le loro confidenze
        results = []
        for idx, confidence in enumerate(predictions[0]):
            results.append({
                'class': class_names[idx],
                'confidence': float(confidence)
            })
        
        # Ordina i risultati per confidenza decrescente
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'predictions': results,
            'top_prediction': results[0]['class']
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Debug
        return jsonify({'error': str(e)}), 400

# Route per gestire le registrazioni dei beta tester
@app.route('/beta-tester', methods=['POST'])
def register_beta_tester():
    try:
        # Ottieni i dati dal form
        data = request.json
        nome = data.get('nome')
        cognome = data.get('cognome')
        email = data.get('email')
        
        # Verifica che tutti i campi siano presenti
        if not all([nome, cognome, email]):
            return jsonify({'success': False, 'error': 'Tutti i campi sono obbligatori'}), 400
        
        # Crea la directory se non esiste
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Percorso del file CSV
        csv_path = os.path.join(data_dir, 'beta_tester.csv')
        
        # Verifica se il file esiste già
        file_exists = os.path.isfile(csv_path)
        
        # Timestamp corrente
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Apri il file in modalità append
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['nome', 'cognome', 'email', 'data_registrazione']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Scrivi l'intestazione solo se il file è nuovo
            if not file_exists:
                writer.writeheader()
            
            # Scrivi i dati
            writer.writerow({
                'nome': nome,
                'cognome': cognome,
                'email': email,
                'data_registrazione': timestamp
            })
        
        # Invia email di conferma
        try:
            send_confirmation_email(nome, cognome, email)
        except Exception as e:
            print(f"Errore nell'invio dell'email: {str(e)}")
            # Non interrompiamo il flusso se l'email fallisce
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        print(f"Errore durante la registrazione del beta tester: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def send_confirmation_email(nome, cognome, email):
    """
    Invia un'email di conferma al beta tester registrato
    """
    try:
        # Configurazione del server SMTP
        smtp_server = app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = app.config.get('SMTP_PORT', 587)
        sender_email = app.config.get('SMTP_USERNAME', 'your-email@gmail.com')  # Sostituire con l'email reale
        sender_password = app.config.get('SMTP_PASSWORD', 'your-app-password')  # Sostituire con la password dell'app
        
        # Creazione del messaggio
        message = MIMEMultipart("alternative")
        message["Subject"] = "Conferma registrazione Beta Tester PL-AI"
        message["From"] = sender_email
        message["To"] = email
        
        # Versione HTML dell'email
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #1e3a8a; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ font-size: 12px; color: #666; text-align: center; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Grazie per esserti registrato come Beta Tester!</h1>
                </div>
                <div class="content">
                    <p>Gentile {nome} {cognome},</p>
                    <p>grazie per esserti registrato come Beta Tester per la piattaforma PL-AI.</p>
                    <p>Ti contatteremo presto con ulteriori informazioni sul programma beta e su come accedere alle funzionalità esclusive.</p>
                    <p>Nel frattempo, se hai domande o suggerimenti, non esitare a contattarci.</p>
                    <p>Cordiali saluti,<br>Il team PL-AI</p>
                </div>
                <div class="footer">
                    <p>Questa è un'email automatica, si prega di non rispondere direttamente a questo messaggio.</p>
                    <p> 2023 PL-AI. Tutti i diritti riservati.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Testo dell'email (versione plain text)
        text = f"""
        Gentile {nome} {cognome},
        
        Grazie per esserti registrato come Beta Tester per la piattaforma PL-AI.
        
        Ti contatteremo presto con ulteriori informazioni sul programma beta e su come accedere alle funzionalità esclusive.
        
        Nel frattempo, se hai domande o suggerimenti, non esitare a contattarci.
        
        Cordiali saluti,
        Il team PL-AI
        """
        
        # Allegare entrambe le versioni
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)
        
        # Connessione al server e invio
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
            
        print(f"Email di conferma inviata a {email}")
        return True
        
    except Exception as e:
        print(f"Errore nell'invio dell'email: {str(e)}")
        raise

# Route per le impostazioni utente
@app.route('/impostazioni')
@login_required
def impostazioni():
    """Pagina delle impostazioni utente"""
    username = session.get('username')
    if not username:
        flash('Devi effettuare il login per accedere alle impostazioni', 'error')
        return redirect(url_for('login'))
    
    # Ottieni i dati dell'utente dal database
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT nome, cognome, email, theme, language FROM users WHERE username = ?', (username,))
    user_data = c.fetchone()
    
    if user_data:
        user = {
            'nome': user_data[0],
            'cognome': user_data[1],
            'email': user_data[2],
            'theme': user_data[3] or 'light',
            'language': user_data[4] or 'it'
        }
    else:
        user = {
            'nome': '',
            'cognome': '',
            'email': '',
            'theme': 'light',
            'language': 'it'
        }
    
    return render_template('impostazioni.html', user=user)

# Route per aggiornare le impostazioni utente
@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    """Aggiorna le impostazioni dell'utente"""
    username = session.get('username')
    if not username:
        flash('Devi effettuare il login per aggiornare le impostazioni', 'error')
        return redirect(url_for('login'))
    
    # Ottieni i dati dal form
    nome = request.form.get('nome', '')
    cognome = request.form.get('cognome', '')
    email = request.form.get('email', '')
    theme = request.form.get('theme', 'light')
    language = request.form.get('language', 'it')
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    # Connessione al database
    conn = get_db()
    c = conn.cursor()
    
    # Aggiorna i dati dell'utente
    c.execute('UPDATE users SET nome = ?, cognome = ?, email = ?, theme = ?, language = ? WHERE username = ?',
              (nome, cognome, email, theme, language, username))
    
    # Gestisci il cambio password se richiesto
    if current_password and new_password and confirm_password:
        # Verifica la password attuale
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        stored_password = c.fetchone()[0]
        
        if stored_password != current_password:
            flash('La password attuale non è corretta', 'error')
            conn.commit()
            return redirect(url_for('impostazioni'))
        
        if new_password != confirm_password:
            flash('Le nuove password non corrispondono', 'error')
            conn.commit()
            return redirect(url_for('impostazioni'))
        
        # Aggiorna la password
        c.execute('UPDATE users SET password = ? WHERE username = ?', (new_password, username))
        flash('Password aggiornata con successo', 'success')
    
    conn.commit()
    conn.close()
    
    flash('Impostazioni aggiornate con successo', 'success')
    return redirect(url_for('impostazioni'))

# Route per eliminare l'account
@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    """Elimina l'account dell'utente"""
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Devi effettuare il login per eliminare l\'account'}), 401
    
    try:
        data = request.get_json()
        password = data.get('password')
        
        if not password:
            return jsonify({'error': 'Password richiesta per confermare l\'eliminazione'}), 400
        
        # Verifica la password
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        
        if not result or result[0] != password:
            return jsonify({'error': 'Password non corretta'}), 401
        
        # Elimina l'utente dal database
        c.execute('DELETE FROM users WHERE username = ?', (username,))
        
        # Elimina anche il database delle lezioni dell'utente
        user_db_path = get_user_db_path(username)
        if os.path.exists(user_db_path):
            try:
                os.remove(user_db_path)
                print(f"Database delle lezioni dell'utente {username} eliminato: {user_db_path}")
            except Exception as e:
                print(f"Errore nell'eliminazione del database delle lezioni: {str(e)}")
        
        conn.commit()
        conn.close()
        
        # Elimina la sessione
        session.clear()
        
        return jsonify({'success': True, 'message': 'Account eliminato con successo'})
        
    except Exception as e:
        print(f"Errore nell'eliminazione dell'account: {str(e)}")
        return jsonify({'error': f'Errore durante l\'eliminazione dell\'account: {str(e)}'}), 500

# Context processor per rendere current_user sempre disponibile nei template
from flask_login import current_user

@app.context_processor
def inject_user():
    return dict(current_user=current_user)

# Route per matematica
@app.route('/matematica')
@login_required
def matematica():
    return render_template('matematica.html')

@app.route('/api/math/progress')
@login_required
def math_progress():
    username = session.get('username')
    conn = get_user_db(username)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM math_progress')
    progress = cursor.fetchall()
    conn.close()
    return jsonify({'progress': progress})

@app.route('/api/math/update-progress', methods=['POST'])
@login_required
def update_math_progress():
    try:
        data = request.get_json()
        topic = data.get('topic')
        completed = data.get('completed', False)
        correct_answers = data.get('correctAnswers', 0)
        
        username = session.get('username')
        conn = get_user_db(username)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO math_progress (topic, completed, correct_answers)
            VALUES (?, ?, ?)
        ''', (topic, completed, correct_answers))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Configurazione per il deployment


@app.route('/machinelearning2')
@login_required
def machinelearning2():
    return render_template('machinelearning2.html')


@app.route('/train_model_unified', methods=['POST'])
@login_required
def train_model_unified():
    try:
        # --- Pulisci i dati del modello precedente dalla sessione ---
        session.pop('model_pickle', None)
        session.pop('label_encoder_pickle', None)
        session.pop('feature_columns', None)
        session.pop('problem_type', None)
        app.logger.info('Puliti i dati del modello precedente dalla sessione.')
        # --- Fine pulizia sessione ---

        data = request.json
        app.logger.info(f"Ricevuti dati per il training: {data}")
        
        raw_data = data.get('rawData')
        target_column = data.get('targetColumn')
        problem_type = data.get('problemType')
        algorithm_name = data.get('algorithmName')
        algorithm_params_frontend = data.get('algorithmParams', {})

        if not all([raw_data, target_column, problem_type, algorithm_name]):
            return jsonify({'error': 'Dati mancanti per il training.'}), 400

        df = pd.DataFrame(raw_data)
        app.logger.info(f"DataFrame creato con colonne: {df.columns.tolist()}")

        if target_column not in df.columns:
            return jsonify({'error': f'Colonna target "{target_column}" non trovata nel dataset.'}), 400

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # --- Gestione Parametri Algoritmo (da convertire e validare) ---
        parsed_params = {}
        # --- Gestione Parametri Algoritmo (da convertire e validare) ---
        parsed_params = {}
        # Definisce come parsare i parametri per ciascun algoritmo
        # Questo dovrebbe rispecchiare la struttura in machinelearning2.js
        # Nota: questa è una gestione semplificata, potrebbe essere necessario un parsing più robusto
        
        def parse_param(value, param_type, default_if_none=None, choices=None):
            if value is None or str(value).strip().lower() == 'none' or str(value).strip() == '':
                return default_if_none # Spesso None per sklearn
            if param_type == 'int':
                return int(value)
            if param_type == 'float':
                return float(value)
            if param_type == 'bool': # Inviato come stringa 'true'/'false' o booleano da JS
                return str(value).lower() == 'true'
            if param_type == 'str_or_float': # per gamma in SVC
                if isinstance(value, str) and value in ['scale', 'auto']:
                    return value
                return float(value)
            if choices and value not in choices:
                 raise ValueError(f"Valore '{value}' non valido. Valori permessi: {choices}")
            return str(value) # Default a stringa se non specificato altrimenti

        try:
            # Logistic Regression
            if algorithm_name == 'Logistic Regression':
                if 'penalty' in algorithm_params_frontend: parsed_params['penalty'] = parse_param(algorithm_params_frontend['penalty'], 'str', default_if_none='l2', choices=['l1', 'l2', 'elasticnet', 'none'])
                if 'C' in algorithm_params_frontend: parsed_params['C'] = parse_param(algorithm_params_frontend['C'], 'float', default_if_none=1.0)
                if 'solver' in algorithm_params_frontend: parsed_params['solver'] = parse_param(algorithm_params_frontend['solver'], 'str', default_if_none='lbfgs') # choices dipendono da penalty
                if 'max_iter' in algorithm_params_frontend: parsed_params['max_iter'] = parse_param(algorithm_params_frontend['max_iter'], 'int', default_if_none=100)
            
            # SVC
            elif algorithm_name == 'SVC':
                if 'C' in algorithm_params_frontend: parsed_params['C'] = parse_param(algorithm_params_frontend['C'], 'float', default_if_none=1.0)
                if 'kernel' in algorithm_params_frontend: parsed_params['kernel'] = parse_param(algorithm_params_frontend['kernel'], 'str', default_if_none='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'])
                if 'degree' in algorithm_params_frontend: parsed_params['degree'] = parse_param(algorithm_params_frontend['degree'], 'int', default_if_none=3) # Solo per kernel='poly'
                if 'gamma' in algorithm_params_frontend: parsed_params['gamma'] = parse_param(algorithm_params_frontend['gamma'], 'str_or_float', default_if_none='scale')
                parsed_params['probability'] = True # Necessario per predict_proba se lo usassimo

            # Decision Tree Classifier
            elif algorithm_name == 'Decision Tree Classifier':
                if 'criterion' in algorithm_params_frontend: parsed_params['criterion'] = parse_param(algorithm_params_frontend['criterion'], 'str', default_if_none='gini', choices=['gini', 'entropy'])
                if 'max_depth' in algorithm_params_frontend: parsed_params['max_depth'] = parse_param(algorithm_params_frontend['max_depth'], 'int', default_if_none=None)
                if 'min_samples_split' in algorithm_params_frontend: parsed_params['min_samples_split'] = parse_param(algorithm_params_frontend['min_samples_split'], 'int', default_if_none=2)
                if 'min_samples_leaf' in algorithm_params_frontend: parsed_params['min_samples_leaf'] = parse_param(algorithm_params_frontend['min_samples_leaf'], 'int', default_if_none=1)

            # Random Forest Classifier
            elif algorithm_name == 'Random Forest Classifier':
                if 'n_estimators' in algorithm_params_frontend: parsed_params['n_estimators'] = parse_param(algorithm_params_frontend['n_estimators'], 'int', default_if_none=100)
                if 'criterion' in algorithm_params_frontend: parsed_params['criterion'] = parse_param(algorithm_params_frontend['criterion'], 'str', default_if_none='gini', choices=['gini', 'entropy'])
                if 'max_depth' in algorithm_params_frontend: parsed_params['max_depth'] = parse_param(algorithm_params_frontend['max_depth'], 'int', default_if_none=None)
                if 'min_samples_split' in algorithm_params_frontend: parsed_params['min_samples_split'] = parse_param(algorithm_params_frontend['min_samples_split'], 'int', default_if_none=2)
                if 'min_samples_leaf' in algorithm_params_frontend: parsed_params['min_samples_leaf'] = parse_param(algorithm_params_frontend['min_samples_leaf'], 'int', default_if_none=1)
            
            # Linear Regression
            elif algorithm_name == 'Linear Regression':
                # Non ha molti parametri sintonizzabili esposti comunemente in questo modo
                pass 

            # Ridge Regression
            elif algorithm_name == 'Ridge Regression':
                if 'alpha' in algorithm_params_frontend: parsed_params['alpha'] = parse_param(algorithm_params_frontend['alpha'], 'float', default_if_none=1.0)

            # Lasso Regression
            elif algorithm_name == 'Lasso Regression':
                if 'alpha' in algorithm_params_frontend: parsed_params['alpha'] = parse_param(algorithm_params_frontend['alpha'], 'float', default_if_none=1.0)
                if 'max_iter' in algorithm_params_frontend: parsed_params['max_iter'] = parse_param(algorithm_params_frontend['max_iter'], 'int', default_if_none=1000)

            # Random Forest Regressor
            elif algorithm_name == 'Random Forest Regressor':
                if 'n_estimators' in algorithm_params_frontend: parsed_params['n_estimators'] = parse_param(algorithm_params_frontend['n_estimators'], 'int', default_if_none=100)
                if 'max_depth' in algorithm_params_frontend: parsed_params['max_depth'] = parse_param(algorithm_params_frontend['max_depth'], 'int', default_if_none=None)
                if 'min_samples_split' in algorithm_params_frontend: parsed_params['min_samples_split'] = parse_param(algorithm_params_frontend['min_samples_split'], 'int', default_if_none=2)
                if 'min_samples_leaf' in algorithm_params_frontend: parsed_params['min_samples_leaf'] = parse_param(algorithm_params_frontend['min_samples_leaf'], 'int', default_if_none=1)

        except ValueError as ve:
            return jsonify({'error': f'Parametro non valido: {str(ve)}'}), 400

        # --- Preprocessing ---
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        # Imputazione per numeriche e categoriche
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()) # Scalare è generalmente una buona pratica
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Crea il preprocessor con ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # scarta colonne non specificate (se ce ne fossero)
        )
        
        # Label Encoding per la variabile target in classificazione
        if problem_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.dtype == 'object': # Regressione con target categorico non ha senso
             # Tenta la conversione se possibile, altrimenti errore
            try:
                y = pd.to_numeric(y)
            except ValueError:
                return jsonify({'error': 'La colonna target per la regressione deve essere numerica o convertibile in numerica.'}), 400

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- Selezione e Addestramento Modello --- 
        model_instance = None
        if problem_type == 'classification':
            if algorithm_name == 'Logistic Regression':
                model_instance = LogisticRegression(**parsed_params, random_state=42)
            elif algorithm_name == 'Linear Discriminant Analysis (LDA)':
                 model_instance = LinearDiscriminantAnalysis(**parsed_params)
            elif algorithm_name == 'Stochastic Gradient Descent Classifier (SGDClassifier)':
                 model_instance = SGDClassifier(**parsed_params, random_state=42)

            elif algorithm_name == 'K-Nearest Neighbors (KNN)':
                 # Aggiungi gestione parametri per KNN Classifier se necessario
                 model_instance = KNeighborsClassifier(**parsed_params)
            elif algorithm_name == 'Decision Tree Classifier':
                model_instance = DecisionTreeClassifier(**parsed_params, random_state=42)
            elif algorithm_name == 'Random Forest Classifier':
                model_instance = RandomForestClassifier(**parsed_params, random_state=42)
            elif algorithm_name == 'Gradient Boosting Classifier (XGBoost)' or \
                 algorithm_name == 'Gradient Boosting Classifier (LightGBM)' or \
                 algorithm_name == 'Gradient Boosting Classifier (CatBoost)':
                 # Nota: XGBoost, LightGBM, CatBoost richiedono le loro librerie. Usiamo scikit-learn's GradientBoostingClassifier come placeholder o aggiungiamo le librerie se disponibili.
                 # Per ora usiamo GradientBoostingClassifier di sklearn come placeholder
                 model_instance = GradientBoostingClassifier(**parsed_params, random_state=42)
            elif algorithm_name == 'Support Vector Machines (SVM)':
                 # Aggiungi gestione parametri kernel etc per SVC
                 model_instance = SVC(**parsed_params, random_state=42)
            elif algorithm_name == 'Naive Bayes (Gaussian)':
                 model_instance = GaussianNB(**parsed_params)
            elif algorithm_name == 'Naive Bayes (Multinomial)':
                 # Richiede che i dati siano conteggi/frequenze, potrebbe necessitare pre-elaborazione specifica
                 model_instance = MultinomialNB(**parsed_params)
            elif algorithm_name == 'Naive Bayes (Bernoulli)':
                 # Richiede dati binari, potrebbe necessitare pre-elaborazione specifica
                 model_instance = BernoulliNB(**parsed_params)
            elif algorithm_name == 'Neural Networks (MLPClassifier)':
                 # Aggiungi gestione parametri per MLPClassifier
                 model_instance = MLPClassifier(**parsed_params, random_state=42, max_iter=1000) # Aumenta max_iter se necessario
            else:
                return jsonify({'error': f'Algoritmo di classificazione "{algorithm_name}" non supportato nel backend.'}), 400
            
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model_instance)])
            
        elif problem_type == 'regression':
            if algorithm_name == 'Linear Regression (OLS)':
                model_instance = LinearRegression(**parsed_params)
            elif algorithm_name == 'Ridge Regression':
                model_instance = Ridge(**parsed_params, random_state=42)
            elif algorithm_name == 'Lasso Regression':
                model_instance = Lasso(**parsed_params, random_state=42)
            elif algorithm_name == 'Elastic Net':
                 model_instance = ElasticNet(**parsed_params, random_state=42)
            elif algorithm_name == 'Bayesian Linear Regression':
                 # Scikit-learn ha BayesianRidge che è una forma di Bayesian Linear Regression
                 model_instance = BayesianRidge(**parsed_params)
            elif algorithm_name == 'SGD Regressor':
                 model_instance = SGDRegressor(**parsed_params, random_state=42)

            elif algorithm_name == 'Polynomial Regression':
                 # Polynomial Regression richiede un pipeline che include PolynomialFeatures e Linear Regression
                 degree = parsed_params.get('degree', 2) # Assumi un parametro 'degree' per il grado polinomiale
                 model_instance = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(**parsed_params))
                 # Nota: Quando si usa make_pipeline così, model_instance è già il pipeline completo.
                 # Dobbiamo gestire questo caso in modo diverso più avanti.
                 model = model_instance # In questo caso, il pipeline è già il modello finale
                 is_pipeline_already_set = True # Flag per saltare la creazione standard del pipeline sotto

            elif algorithm_name == 'Support Vector Regression (SVR)':
                 # Aggiungi gestione parametri kernel etc per SVR
                 model_instance = SVR(**parsed_params)
            elif algorithm_name == 'Decision Tree Regressor':
                model_instance = DecisionTreeRegressor(**parsed_params, random_state=42)
            elif algorithm_name == 'Random Forest Regressor':
                model_instance = RandomForestRegressor(**parsed_params, random_state=42)
            elif algorithm_name == 'Gradient Boosting Regressor (XGBoost)' or \
                 algorithm_name == 'Gradient Boosting Regressor (LightGBM)' or \
                 algorithm_name == 'Gradient Boosting Regressor (CatBoost)':
                 # Usiamo scikit-learn's GradientBoostingRegressor come placeholder
                 model_instance = GradientBoostingRegressor(**parsed_params, random_state=42)
            elif algorithm_name == 'K-Nearest Neighbors Regressor':
                 # Aggiungi gestione parametri per KNN Regressor se necessario
                 model_instance = KNeighborsRegressor(**parsed_params)
            elif algorithm_name == 'Neural Networks (MLPRegressor)':
                 # Aggiungi gestione parametri per MLPRegressor
                 model_instance = MLPRegressor(**parsed_params, random_state=42, max_iter=1000) # Aumenta max_iter se necessario
            else:
                return jsonify({'error': f'Algoritmo di regressione "{algorithm_name}" non supportato nel backend.'}), 400

            # Gestisci il caso speciale di Polynomial Regression dove il pipeline è già stato creato
            if algorithm_name != 'Polynomial Regression':
                 model = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model_instance)])
            # Altrimenti, 'model' è già stato assegnato nel blocco Polynomial Regression

        else:
            return jsonify({'error': f'Tipo di problema "{problem_type}" non supportato.'}), 400

        # --- Addestramento ---
        app.logger.info('Inizio fase di addestramento.')
        app.logger.info(f'Tipo di oggetto model prima del fit: {type(model)}')
        # Per Polynomial Regression, il fit è già incluso nel make_pipeline
        if algorithm_name != 'Polynomial Regression':
             app.logger.info('Chiamata model.fit()...')
             model.fit(X_train, y_train)
             app.logger.info('Chiamata model.fit() completata.')
        else:
             app.logger.info('Polynomial Regression: fit incluso nel make_pipeline, saltando model.fit().')

        y_pred = model.predict(X_test)

        # --- Calcolo Metriche ---
        metrics = {}
        if problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            # Confusion matrix potrebbe essere utile, ma più complessa da inviare/visualizzare
            # cm = confusion_matrix(y_test, y_pred)
            # metrics['confusion_matrix'] = cm.tolist() # Converti a lista per JSON
        
        elif problem_type == 'regression':
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)

        # Prepara le informazioni sulle feature per la predizione
        feature_info = {}
        for feature in X.columns:
            if pd.api.types.is_numeric_dtype(X[feature]):
                feature_info[feature] = {
                    'type': 'numeric',
                    'min': float(X[feature].min()),
                    'max': float(X[feature].max()),
                    'default': float(X[feature].mean())
                }
            else:
                feature_info[feature] = {
                    'type': 'categorical',
                    'values': X[feature].unique().tolist()
                }

        # Serialize the model and label encoder (if classification)
        model_pickle = pickle.dumps(model)
        label_encoder_pickle = pickle.dumps(le) if problem_type == 'classification' else None

        # Store the serialized objects and other necessary info in the session
        session['model_pickle'] = model_pickle
        session['label_encoder_pickle'] = label_encoder_pickle
        session['feature_columns'] = X.columns.tolist()
        session['problem_type'] = problem_type

        response_data = {
            'metrics': metrics,
            'feature_info': feature_info,
            'message': 'Training completato con successo!'
        }
        app.logger.info(f"Risposta preparata: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Errore in /train_model_unified: {str(e)}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Errore interno del server: {str(e)}'}), 500


# RAG (Retrieval-Augmented Generation) Routes

def init_rag_database(db_path):
    """Inizializza la tabella rag_documents nel database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rag_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        size INTEGER NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT 0,
        text_content TEXT,
        embedding_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/rag')
@login_required
def rag():
    """Renderizza la pagina RAG"""
    # Inizializza il database se necessario
    db_path = get_user_db_path(session.get('username'))
    init_rag_database(db_path)
    return render_template('rag.html')

# Function to ensure columns exist in rag_documents table
def migrate_rag_documents_table(username):
    """Ensure rag_documents table has all required columns."""
    try:
        db_path = get_user_db_path(username) 
        if not os.path.exists(db_path):
            return False  # Database doesn't exist yet
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_documents'")
        if not cursor.fetchone():
            conn.close()
            return False  # Table doesn't exist yet
        
        # Check and add user_id column if missing
        cursor.execute("PRAGMA table_info(rag_documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        altered = False
        if 'user_id' not in columns:
            print(f"Adding user_id column to rag_documents for {username}")
            cursor.execute("ALTER TABLE rag_documents ADD COLUMN user_id INTEGER")
            altered = True
            
        if 'processed_for_lesson' not in columns:
            print(f"Adding processed_for_lesson column to rag_documents for {username}")
            cursor.execute("ALTER TABLE rag_documents ADD COLUMN processed_for_lesson BOOLEAN DEFAULT 0")
            altered = True
            
        if altered:
            # Backfill user_id for existing documents based on username
            user_id = session.get('user_id')
            if user_id:
                cursor.execute("UPDATE rag_documents SET user_id = ? WHERE user_id IS NULL", (user_id,))
                print(f"Backfilled user_id={user_id} for existing rag_documents")
            
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error migrating rag_documents table: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/upload_rag_document', methods=['POST'])
@login_required
def upload_rag_document():
    """Endpoint per caricare documenti per RAG"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Nessun file inviato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nessun file selezionato'}), 400
    
    # Crea directory per documenti RAG dell'utente se non esiste
    user_id = session.get('user_id')
    username = session.get('username')
    user_rag_dir = os.path.join('user_data', str(user_id), 'rag_documents')
    os.makedirs(user_rag_dir, exist_ok=True)
    
    # Salva il file con nome sicuro
    filename = secure_filename(file.filename)
    file_path = os.path.join(user_rag_dir, filename)
    
    # Controlla se il file esiste già e aggiungi un suffisso numerico se necessario
    base_name, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(file_path):
        filename = f"{base_name}_{counter}{extension}"
        file_path = os.path.join(user_rag_dir, filename)
        counter += 1
    
    try:
        # Salva il file
        file.save(file_path)
        
        # Salva informazioni sul file nel database dell'utente
        db_path = get_user_db_path(username)
        
        # Migrazione: assicura che la tabella rag_documents abbia le colonne necessarie
        migrate_rag_documents_table(username)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Crea tabella se non esiste - con nuovo schema che include user_id
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            size INTEGER NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT 0,
            text_content TEXT,
            embedding_path TEXT,
            user_id INTEGER,
            processed_for_lesson BOOLEAN DEFAULT 0
        )
        ''')
        
        # Determina il tipo di file
        file_type = extension.lower().replace('.', '')
        file_size = os.path.getsize(file_path)
        
        # Tenta l'inserimento con user_id
        try:
            cursor.execute('''
            INSERT INTO rag_documents (filename, file_path, file_type, size, user_id)
            VALUES (?, ?, ?, ?, ?)
            ''', (filename, file_path, file_type, file_size, user_id))
        except sqlite3.OperationalError as e:
            # Se c'è un errore di colonna (nonostante la migrazione), fallback all'inserimento senza user_id
            if 'no column named user_id' in str(e):
                print(f"Warning: Fallback to insertion without user_id column for {username}")
                cursor.execute('''
                INSERT INTO rag_documents (filename, file_path, file_type, size)
                VALUES (?, ?, ?, ?)
                ''', (filename, file_path, file_type, file_size))
            else:
                # Rilancia altri errori SQLite
                raise
        
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'file_id': file_id
        }), 200
        
    except Exception as e:
        print(f"Errore durante il caricamento del file: {str(e)}")
        # Se c'è un errore, elimina il file se è stato creato
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({
            'success': False, 
            'error': f'Errore durante il caricamento del file: {str(e)}'
        }), 500

@app.route('/process_rag_documents', methods=['POST'])
@login_required
def process_rag_documents():
    """Avvia il processo di elaborazione dei documenti caricati"""
    try:
        user_id = session.get('user_id')
        print(f"[DEBUG] Starting document processing for user {user_id}")
        
        # Crea directory per i file elaborati se non esiste
        user_rag_processed_dir = os.path.join('user_data', str(user_id), 'rag_processed')
        os.makedirs(user_rag_processed_dir, exist_ok=True)
        
        # Inizializza lo stato di elaborazione nella sessione
        session['rag_processing'] = {
            'status': 'starting',
            'progress': 0,
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'current_document': '',
            'errors': []
        }
        
        # Ottieni i documenti non elaborati dal database
        db_path = get_user_db_path(session.get('username'))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, filename, file_path, file_type FROM rag_documents 
        WHERE processed = 0
        ''')
        
        documents = cursor.fetchall()
        conn.close()
        
        print(f"[DEBUG] Found {len(documents)} documents to process")
        
        # Aggiorna lo stato di elaborazione
        session['rag_processing']['total_documents'] = len(documents)
        session.modified = True
        
        if len(documents) == 0:
            print("[DEBUG] No documents to process")
            return jsonify({'success': False, 'error': 'Nessun documento da elaborare'}), 400
        
        # Avvia l'estrazione del testo
        print("[DEBUG] Starting text extraction")
        extract_text_from_documents()
        
        return jsonify({
            'success': True, 
            'message': 'Elaborazione documenti avviata', 
            'document_count': len(documents)
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Error in process_rag_documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Errore durante l\'elaborazione dei documenti: {str(e)}'
        }), 500

@app.route('/check_text_extraction_status')
@login_required
def check_text_extraction_status():
    """Controlla lo stato dell'estrazione del testo dai documenti"""
    # In un'implementazione reale, questo dovrebbe controllare lo stato di un task asincrono
    # Per questa demo, simuliamo l'avanzamento dell'estrazione del testo
    
    processing_state = session.get('rag_processing', {
        'status': 'not_started',
        'progress': 0,
        'total_documents': 0,
        'processed_documents': 0,
        'failed_documents': 0,
        'current_document': '',
        'errors': []
    })
    
    print(f"[DEBUG] Current processing state: {processing_state}")
    
    # Se non è stato avviato, avvia l'estrazione del testo
    if processing_state['status'] == 'starting':
        print("[DEBUG] Starting text extraction process")
        # Simuliamo l'avvio dell'estrazione del testo
        processing_state['status'] = 'extracting_text'
        processing_state['progress'] = 10
        session['rag_processing'] = processing_state
        session.modified = True
        
        # Avvia l'estrazione del testo (simulata)
        extract_text_from_documents()
        
        return jsonify({
            'status_message': 'Avvio estrazione testo...',
            'progress': 10,
            'completed': False,
            'success': True
        })
    
    # Se l'estrazione è in corso, restituisci lo stato corrente
    elif processing_state['status'] == 'extracting_text':
        total = processing_state['total_documents']
        processed = processing_state['processed_documents']
        failed = processing_state['failed_documents']
        current = processing_state['current_document']
        
        print(f"[DEBUG] Extraction in progress - Total: {total}, Processed: {processed}, Failed: {failed}, Current: {current}")
        
        # Calcola la percentuale di avanzamento (dal 10% al 50%)
        if total > 0:
            base_progress = 10
            extraction_progress = int(40 * (processed + failed) / total)
            progress = min(base_progress + extraction_progress, 50)
        else:
            progress = 10
        
        print(f"[DEBUG] Calculated progress: {progress}%")
        
        processing_state['progress'] = progress
        session['rag_processing'] = processing_state
        session.modified = True
        
        # Controlla se l'estrazione è completata
        completed = (processed + failed) >= total
        success = failed < total  # Consideriamo il processo riuscito se almeno un documento è stato elaborato con successo
        
        if completed:
            print("[DEBUG] Text extraction completed")
            processing_state['status'] = 'text_extraction_completed'
            session['rag_processing'] = processing_state
            session.modified = True
        
        return jsonify({
            'status_message': f"Estrazione testo: {processed}/{total} documenti" + (f" (Elaborazione: {current})" if current else ""),
            'progress': progress,
            'completed': completed,
            'success': success,
            'errors': processing_state['errors'] if failed > 0 else []
        })
    
    # Se l'estrazione è completata
    elif processing_state['status'] == 'text_extraction_completed':
        print("[DEBUG] Text extraction already completed")
        return jsonify({
            'status_message': 'Estrazione testo completata',
            'progress': 50,
            'completed': True,
            'success': processing_state['failed_documents'] < processing_state['total_documents']
        })
    
    # Se non è stato avviato
    else:
        print("[DEBUG] Text extraction not started")
        return jsonify({
            'status_message': 'Estrazione testo non avviata',
            'progress': 0,
            'completed': False,
            'success': False
        })

def extract_text_from_documents():
    """Estrae il testo dai documenti caricati"""
    user_id = session.get('user_id')
    db_path = get_user_db_path(session.get('username'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"[DEBUG] Starting text extraction for user {user_id}")
    
    # Ottieni i documenti non elaborati
    cursor.execute('''
    SELECT id, filename, file_path, file_type FROM rag_documents 
    WHERE processed = 0
    ''')
    
    documents = cursor.fetchall()
    print(f"[DEBUG] Found {len(documents)} documents to process")
    
    for doc_id, filename, file_path, file_type in documents:
        try:
            print(f"[DEBUG] Processing document: {filename} (ID: {doc_id}, Type: {file_type})")
            
            # Aggiorna lo stato di elaborazione
            processing_state = session.get('rag_processing', {})
            processing_state['current_document'] = filename
            session['rag_processing'] = processing_state
            session.modified = True
            
            text_content = ""
            
            # Estrai il testo in base al tipo di file
            if file_type in ['txt']:
                print(f"[DEBUG] Extracting text from TXT file")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                print(f"[DEBUG] Extracted {len(text_content)} characters from TXT")
            
            elif file_type in ['pdf']:
                print(f"[DEBUG] Extracting text from PDF file")
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    print(f"[DEBUG] PDF has {num_pages} pages")
                    text_content = ""
                    for i, page in enumerate(pdf_reader.pages):
                        print(f"[DEBUG] Processing page {i+1}/{num_pages}")
                        page_text = page.extract_text() or ""
                        text_content += page_text + "\n"
                        print(f"[DEBUG] Extracted {len(page_text)} characters from page {i+1}")
                print(f"[DEBUG] Total extracted text length: {len(text_content)} characters")
            
            elif file_type in ['doc', 'docx']:
                print(f"[DEBUG] Extracting text from DOC/DOCX file")
                import docx
                doc = docx.Document(file_path)
                text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                print(f"[DEBUG] Extracted {len(text_content)} characters from DOC/DOCX")
            
            elif file_type in ['jpg', 'jpeg', 'png']:
                print(f"[DEBUG] Extracting text from image file")
                import pytesseract
                from PIL import Image
                image = Image.open(file_path)
                print(f"[DEBUG] Image size: {image.size}, format: {image.format}")
                text_content = pytesseract.image_to_string(image, lang='ita+eng')
                print(f"[DEBUG] Extracted {len(text_content)} characters from image")
            
            print(f"[DEBUG] Saving extracted text to database for document {doc_id}")
            # Salva il testo estratto nel database
            cursor.execute('''
            UPDATE rag_documents 
            SET text_content = ?, processed = 1
            WHERE id = ?
            ''', (text_content, doc_id))
            
            conn.commit()
            print(f"[DEBUG] Successfully saved text to database for document {doc_id}")
            
            # Aggiorna il contatore dei documenti elaborati
            processing_state = session.get('rag_processing', {})
            processing_state['processed_documents'] += 1
            session['rag_processing'] = processing_state
            session.modified = True
            print(f"[DEBUG] Updated processing state: {processing_state['processed_documents']} documents processed")
            
        except Exception as e:
            print(f"[ERROR] Error processing document {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Aggiorna il contatore dei documenti falliti
            processing_state = session.get('rag_processing', {})
            processing_state['failed_documents'] += 1
            processing_state['errors'].append(f"Errore nel documento {filename}: {str(e)}")
            session['rag_processing'] = processing_state
            session.modified = True
            print(f"[DEBUG] Updated processing state: {processing_state['failed_documents']} documents failed")
    
    print("[DEBUG] Text extraction process completed")
    conn.close()

@app.route('/create_rag_embeddings', methods=['POST'])
@login_required
def create_rag_embeddings():
    """Avvia il processo di creazione degli embedding"""
    try:
        print("[DEBUG] Starting embedding creation process")
        # Inizializza lo stato di elaborazione nella sessione
        session['rag_embedding'] = {
            'status': 'starting',
            'progress': 0,
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'errors': []
        }
        
        # Ottieni i documenti da processare
        db_path = get_user_db_path(session.get('username'))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, text_content FROM rag_documents 
        WHERE processed = 1 AND embedding_path IS NULL
        ''')
        
        documents = cursor.fetchall()
        conn.close()
        
        print(f"[DEBUG] Found {len(documents)} documents for embedding creation")
        
        # Aggiorna lo stato
        session['rag_embedding']['total_documents'] = len(documents)
        session['rag_embedding']['status'] = 'creating_embeddings'
        session.modified = True
        
        if len(documents) == 0:
            print("[DEBUG] No documents to create embeddings for")
            return jsonify({'success': False, 'error': 'Nessun documento da elaborare'}), 400
        
        # Avvia il processo di creazione degli embedding
        create_embeddings_for_documents()
        
        return jsonify({'success': True, 'message': 'Creazione embedding avviata'}), 200
        
    except Exception as e:
        print(f"[ERROR] Error in create_rag_embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Errore durante la creazione degli embedding: {str(e)}'
        }), 500

@app.route('/check_embedding_status')
@login_required
def check_embedding_status():
    """Controlla lo stato della creazione degli embedding"""
    try:
        embedding_state = session.get('rag_embedding', {
            'status': 'not_started',
            'progress': 0,
            'total_documents': 0,
            'processed_documents': 0,
            'failed_documents': 0,
            'errors': []
        })
        
        print(f"[DEBUG] Current embedding state: {embedding_state}")
        
        # Se non è stato avviato
        if embedding_state['status'] == 'not_started':
            print("[DEBUG] Embedding process not started")
            return jsonify({
                'status_message': 'Creazione embedding non avviata',
                'progress': 0,
                'completed': False,
                'success': False
            })
        
        # Se è in corso
        elif embedding_state['status'] == 'creating_embeddings':
            total = embedding_state['total_documents']
            processed = embedding_state['processed_documents']
            failed = embedding_state['failed_documents']
            
            print(f"[DEBUG] Embedding in progress - Total: {total}, Processed: {processed}, Failed: {failed}")
            
            # Calcola la percentuale di avanzamento (dal 60% al 100%)
            if total > 0:
                base_progress = 60
                embedding_progress = int(40 * (processed + failed) / total)
                progress = min(base_progress + embedding_progress, 100)
            else:
                progress = 60
            
            print(f"[DEBUG] Calculated progress: {progress}%")
            
            # Controlla se la creazione è completata
            completed = (processed + failed) >= total
            success = failed < total
            
            if completed:
                print("[DEBUG] Embedding creation completed")
                embedding_state['status'] = 'completed'
                session['rag_embedding'] = embedding_state
                session.modified = True
            
            return jsonify({
                'status_message': f"Creazione embedding: {processed}/{total} documenti",
                'progress': progress,
                'completed': completed,
                'success': success,
                'errors': embedding_state['errors'] if failed > 0 else []
            })
        
        # Se è completata
        elif embedding_state['status'] == 'completed':
            print("[DEBUG] Embedding creation already completed")
            return jsonify({
                'status_message': 'Creazione embedding completata',
                'progress': 100,
                'completed': True,
                'success': True
            })
        
        # Stato non riconosciuto
        else:
            print(f"[DEBUG] Unknown embedding state: {embedding_state['status']}")
            return jsonify({
                'status_message': 'Stato non riconosciuto',
                'progress': 0,
                'completed': False,
                'success': False
            })
            
    except Exception as e:
        print(f"[ERROR] Error in check_embedding_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status_message': f'Errore durante il controllo dello stato: {str(e)}',
            'progress': 0,
            'completed': False,
            'success': False
        }), 500

def create_embeddings_for_documents():
    """Crea gli embedding per i documenti processati"""
    try:
        print("[DEBUG] Starting embedding creation process")
        # Ottieni i documenti da processare
        db_path = get_user_db_path(session.get('username'))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, text_content FROM rag_documents 
        WHERE processed = 1 AND embedding_path IS NULL
        ''')
        
        documents = cursor.fetchall()
        print(f"[DEBUG] Found {len(documents)} documents for embedding creation")
        
        if not documents:
            print("[DEBUG] No documents to create embeddings for")
            return
        
        # Aggiorna lo stato iniziale
        session['rag_embedding'] = {
            'status': 'creating_embeddings',
            'progress': 0,
            'total_documents': len(documents),
            'processed_documents': 0,
            'failed_documents': 0,
            'errors': []
        }
        session.modified = True
        
        # Crea la directory per gli embedding se non esiste
        embeddings_dir = os.path.join(os.path.dirname(db_path), 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Processa ogni documento
        for doc_id, text_content in documents:
            try:
                print(f"[DEBUG] Creating embedding for document {doc_id}")
                # Crea l'embedding
                embedding = create_embedding(text_content)
                
                # Salva l'embedding
                embedding_path = os.path.join(embeddings_dir, f'doc_{doc_id}.npy')
                np.save(embedding_path, embedding)
                
                # Aggiorna il database
                cursor.execute('''
                UPDATE rag_documents 
                SET embedding_path = ? 
                WHERE id = ?
                ''', (embedding_path, doc_id))
                conn.commit()
                
                # Aggiorna lo stato
                session['rag_embedding']['processed_documents'] += 1
                session.modified = True
                print(f"[DEBUG] Successfully created embedding for document {doc_id}")
                
            except Exception as e:
                print(f"[ERROR] Error creating embedding for document {doc_id}: {str(e)}")
                session['rag_embedding']['failed_documents'] += 1
                session['rag_embedding']['errors'].append(f"Errore documento {doc_id}: {str(e)}")
                session.modified = True
        
        # Aggiorna lo stato finale
        session['rag_embedding']['status'] = 'completed'
        session.modified = True
        print("[DEBUG] Embedding creation process completed")
        
    except Exception as e:
        print(f"[ERROR] Error in create_embeddings_for_documents: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'rag_embedding' in session:
            session['rag_embedding']['status'] = 'error'
            session['rag_embedding']['errors'].append(str(e))
            session.modified = True
    finally:
        if 'conn' in locals():
            conn.close()

def create_embedding(text):
    """Crea un embedding per il testo fornito"""
    try:
        print("[DEBUG] Creating embedding for text")
        # Usa il modello per creare l'embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text)
        print(f"[DEBUG] Successfully created embedding of size {embedding.shape}")
        return embedding
    except Exception as e:
        print(f"[ERROR] Error creating embedding: {str(e)}")
        raise

@app.route('/get_rag_documents')
@login_required
def get_rag_documents():
    """Ottiene la lista dei documenti nella knowledge base"""
    db_path = get_user_db_path(session.get('username'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ottieni tutti i documenti elaborati
    cursor.execute('''
    SELECT id, filename, file_type, size, upload_date FROM rag_documents 
    WHERE processed = 1
    ''')
    
    documents = cursor.fetchall()
    conn.close()
    
    # Formatta i risultati
    document_list = []
    for doc_id, filename, file_type, size, upload_date in documents:
        document_list.append({
            'id': doc_id,
            'filename': filename,
            'type': file_type,
            'size': size,
            'upload_date': upload_date
        })
    
    return jsonify({'success': True, 'documents': document_list}), 200

@app.route('/delete_rag_document/<int:doc_id>', methods=['DELETE'])
@login_required
def delete_rag_document(doc_id):
    """Elimina un documento dalla knowledge base"""
    db_path = get_user_db_path(session.get('username'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ottieni il percorso del file e dell'embedding
    cursor.execute('''
    SELECT file_path, embedding_path FROM rag_documents 
    WHERE id = ?
    ''', (doc_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        return jsonify({'success': False, 'error': 'Documento non trovato'}), 404
    
    file_path, embedding_path = result
    
    # Elimina il file se esiste
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
    
    # Elimina l'embedding se esiste
    if embedding_path and os.path.exists(embedding_path):
        os.remove(embedding_path)
    
    # Elimina il record dal database
    cursor.execute('''
    DELETE FROM rag_documents 
    WHERE id = ?
    ''', (doc_id,))
    
    conn.commit()
    
    # Controlla se la knowledge base è vuota
    cursor.execute('''
    SELECT COUNT(*) FROM rag_documents 
    WHERE processed = 1
    ''')
    
    count = cursor.fetchone()[0]
    empty_kb = count == 0
    
    conn.close()
    
    return jsonify({'success': True, 'empty_kb': empty_kb}), 200

@app.route('/clear_rag_knowledge_base', methods=['POST'])
@login_required
def clear_rag_knowledge_base():
    """Cancella l'intera knowledge base"""
    user_id = session.get('user_id')
    db_path = get_user_db_path(session.get('username'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ottieni tutti i percorsi dei file e degli embedding
    cursor.execute('''
    SELECT file_path, embedding_path FROM rag_documents
    ''')
    
    paths = cursor.fetchall()
    
    # Elimina tutti i file e gli embedding
    for file_path, embedding_path in paths:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        if embedding_path and os.path.exists(embedding_path):
            os.remove(embedding_path)
    
    # Elimina tutti i record dal database
    cursor.execute('''
    DELETE FROM rag_documents
    ''')
    
    conn.commit()
    conn.close()
    
    # Elimina le directory se vuote
    user_rag_dir = os.path.join('user_data', str(user_id), 'rag_documents')
    user_embeddings_dir = os.path.join('user_data', str(user_id), 'rag_embeddings')
    
    try:
        if os.path.exists(user_rag_dir) and not os.listdir(user_rag_dir):
            os.rmdir(user_rag_dir)
        
        if os.path.exists(user_embeddings_dir) and not os.listdir(user_embeddings_dir):
            os.rmdir(user_embeddings_dir)
    except Exception as e:
        print(f"Errore nella pulizia delle directory: {str(e)}")
    
    return jsonify({'success': True}), 200

@app.route('/check_rag_knowledge_base')
@login_required
def check_rag_knowledge_base():
    """Controlla se esiste una knowledge base per l'utente"""
    db_path = get_user_db_path(session.get('username'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Controlla se esistono documenti elaborati
    cursor.execute('''
    SELECT COUNT(*) FROM rag_documents 
    WHERE processed = 1
    ''')
    
    try:
        count = cursor.fetchone()[0]
        exists = count > 0
    except:
        # La tabella potrebbe non esistere
        exists = False
    
    conn.close()
    
    return jsonify({'ready': exists}), 200

@app.route('/rag_chat', methods=['POST'])
@login_required
def rag_chat():
    """Gestisce le richieste di chat con la knowledge base"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Messaggio mancante'}), 400
    
    user_message = data['message']
    
    try:
        print("[DEBUG] Starting chat processing")
        # Crea l'embedding del messaggio dell'utente usando SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(user_message)
        print("[DEBUG] Created query embedding")
        
        # Cerca i documenti più rilevanti
        db_path = get_user_db_path(session.get('username'))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, text_content, embedding_path FROM rag_documents WHERE embedding_path IS NOT NULL')
        documents = cursor.fetchall()
        conn.close()
        
        print(f"[DEBUG] Found {len(documents)} documents with embeddings")
        
        # Calcola la similarità con ogni documento
        similarities = []
        for doc_id, text_content, embedding_path in documents:
            try:
                doc_embedding = np.load(embedding_path)
                similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                similarities.append((doc_id, text_content, similarity))
                print(f"[DEBUG] Calculated similarity for document {doc_id}: {similarity}")
            except Exception as e:
                print(f"[ERROR] Error calculating similarity for document {doc_id}: {str(e)}")
                continue
        
        # Ordina per similarità e prendi i primi 3 documenti
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_docs = similarities[:3]
        
        print(f"[DEBUG] Selected top {len(top_docs)} documents")
        
        # Funzione per dividere il testo in chunks
        def split_text_into_chunks(text, max_chunk_size=4000):
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 per lo spazio
                
                if current_size >= max_chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
        
        # Prepara il contesto per il modello
        context_chunks = []
        for i, doc in enumerate(top_docs):
            doc_chunks = split_text_into_chunks(doc[1])
            for chunk in doc_chunks:
                context_chunks.append(f"Documento {i+1} (parte):\n{chunk}")
        
        # Se abbiamo troppi chunks, prendiamo solo i più rilevanti
        if len(context_chunks) > 3:
            context_chunks = context_chunks[:3]
        
        context = "\n\n".join(context_chunks)
        
        # Genera la risposta usando GPT-4 con streaming
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """Sei un assistente esperto che risponde alle domande basandosi sul contesto fornito. 
                Il tuo compito è:
                1. Analizzare attentamente i documenti forniti
                2. Rispondere alla domanda dell'utente basandoti SOLO sulle informazioni presenti nei documenti
                3. Se la risposta non può essere dedotta dal contesto, dillo chiaramente
                4. Se ci sono informazioni contraddittorie nei documenti, segnalalo
                5. Cita i documenti specifici quando possibile
                6. Fornisci risposte dettagliate e precise"""},
                {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda: {user_message}"}
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=True  # Abilita lo streaming
        )
        
        print("[DEBUG] Generated streaming response from GPT-4")

        def generate():
            for chunk in response:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        print(f"[ERROR] Error in rag_chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Errore nella generazione della risposta'}), 500

if __name__ == '__main__':
    # In ambiente di sviluppo
    app.run(debug=True)

@app.route('/api/generate-exercise', methods=['POST'])
@login_required
def generate_exercise():
    try:
        data = request.get_json()
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        
        prompt = f"""Genera un esercizio di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}

Formato richiesto:
1. Domanda chiara in italiano
2. Formule in LaTeX (es: \\(x^2 + 3x = 0\\))
3. Risposta corretta tra [RISPOSTA]...[/RISPOSTA]"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un tutor di matematica esperto. Genera esercizi appropriati per il livello scolastico indicato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        exercise = content.split('[RISPOSTA]')[0].strip()
        answer = content.split('[RISPOSTA]')[1].split('[/RISPOSTA]')[0].strip()
        
        return jsonify({
            'exercise': exercise,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-exercise', methods=['POST'])
@login_required
def generate_exercise():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nessun dato ricevuto'}), 400
            
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        
        if not all([grade, subject, topic]):
            return jsonify({'error': 'Dati mancanti'}), 400

        prompt = f"""Genera un esercizio di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}

Formato richiesto:
1. Domanda chiara in italiano
2. Formule in LaTeX (es: \\(x^2 + 3x = 0\\))
3. Risposta corretta tra [RISPOSTA]...[/RISPOSTA]"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un tutor di matematica esperto. Genera esercizi appropriati per il livello scolastico indicato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        if '[RISPOSTA]' not in content or '[/RISPOSTA]' not in content:
            return jsonify({'error': 'Formato risposta non valido'}), 400
            
        exercise = content.split('[RISPOSTA]')[0].strip()
        answer = content.split('[RISPOSTA]')[1].split('[/RISPOSTA]')[0].strip()
        
        return jsonify({
            'status': 'success',
            'exercise': exercise,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate-lesson', methods=['POST'])
@login_required
def generate_lesson():
    try:
        data = request.get_json()
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        
        prompt = f"""Genera una mini-lezione di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}

La lezione deve:
1. Essere concisa (max 15 righe)
2. Spiegare i concetti in modo chiaro e didattico
3. Includere esempi pratici
4. Usare formule in LaTeX quando necessario (es: \\(x^2 + 3x = 0\\))
5. Essere adatta al livello scolastico indicato"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un insegnante di matematica esperto. Scrivi lezioni chiare e didattiche."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        lesson = response.choices[0].message.content.strip()
        
        return jsonify({
            'status': 'success',
            'lesson': lesson
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate-exercise', methods=['POST'])
@login_required
def generate_exercise():
    try:
        data = request.get_json()
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        exercise_type = data.get('exerciseType')
        
        prompt = f"""Genera un esercizio di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}
- Tipo: {exercise_type}

Formato richiesto:
1. Domanda chiara in italiano
2. Formule in LaTeX (es: \\(x^2 + 3x = 0\\))
3. Risposta corretta tra [RISPOSTA]...[/RISPOSTA]

Per il tipo di esercizio:
- 'fill': domanda con risposta numerica o algebrica
- 'text': domanda che richiede una spiegazione in italiano
- 'image': domanda che richiede di mostrare i passaggi della soluzione"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un tutor di matematica esperto. Genera esercizi appropriati per il livello scolastico indicato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        exercise = content.split('[RISPOSTA]')[0].strip()
        answer = content.split('[RISPOSTA]')[1].split('[/RISPOSTA]')[0].strip()
        
        return jsonify({
            'status': 'success',
            'exercise': exercise,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/check-answer', methods=['POST'])
@login_required
def check_answer():
    try:
        data = request.get_json()
        answer = data.get('answer')
        exercise_type = data.get('exerciseType')
        correct_answer = data.get('correctAnswer')
        
        if exercise_type == 'fill':
            # Per risposte numeriche/algebriche, normalizza e confronta
            is_correct = normalize_answer(answer) == normalize_answer(correct_answer)
            feedback = "Risposta corretta!" if is_correct else f"Risposta errata. La soluzione corretta è: {correct_answer}"
        
        elif exercise_type == 'text':
            # Per risposte testuali, usa GPT per valutare
            prompt = f"""Valuta se la risposta dello studente è corretta rispetto alla soluzione attesa.
            
Soluzione attesa:
{correct_answer}

Risposta dello studente:
{answer}

Rispondi con un JSON nel formato:
{{
    "is_correct": true/false,
    "feedback": "spiegazione dettagliata"
}}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un insegnante di matematica che valuta le risposte degli studenti."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            is_correct = result['is_correct']
            feedback = result['feedback']
        
        else:
            return jsonify({'status': 'error', 'message': 'Tipo di esercizio non supportato'}), 400
        
        return jsonify({
            'status': 'success',
            'isCorrect': is_correct,
            'feedback': feedback
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/check-image-answer', methods=['POST'])
@login_required
def check_image_answer():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'Nessuna immagine caricata'}), 400
        
        image = request.files['image']
        correct_answer = request.form.get('correctAnswer')
        
        # Salva temporaneamente l'immagine
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + secure_filename(image.filename))
        image.save(temp_path)
        
        try:
            # Usa GPT-4 Vision per analizzare l'immagine
            with open(temp_path, 'rb') as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            prompt = f"""Analizza l'immagine della soluzione di un esercizio di matematica e verifica se è corretta.
            
Soluzione attesa:
{correct_answer}

Rispondi con un JSON nel formato:
{{
    "is_correct": true/false,
    "feedback": "spiegazione dettagliata"
}}"""

            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return jsonify({
                'status': 'success',
                'isCorrect': result['is_correct'],
                'feedback': result['feedback']
            })
            
        finally:
            # Pulisci il file temporaneo
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def normalize_answer(answer):
    """Normalizza una risposta matematica per il confronto."""
    # Rimuovi spazi extra
    answer = ' '.join(answer.split())
    
    # Converti in minuscolo
    answer = answer.lower()
    
    # Sostituisci caratteri speciali
    replacements = {
        '×': '*',
        '÷': '/',
        '²': '^2',
        '³': '^3',
        '√': 'sqrt',
        'π': 'pi',
        '∞': 'inf'
    }
    
    for old, new in replacements.items():
        answer = answer.replace(old, new)
    
    return answer
