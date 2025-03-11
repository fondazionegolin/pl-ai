from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
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
from routes.db import login_required

# Import dei blueprint e funzioni di autenticazione
from routes.chatbot import chatbot
from routes.resources import resources

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
from routes.db import get_db, get_user_db, register_user, authenticate_user, login_required, update_api_credits, get_api_credits

# Configurazione dell'applicazione
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chiave_segreta_predefinita')
app.config['USER_DB_DIR'] = os.path.join(os.path.dirname(__file__), 'user_databases')

# Assicurati che la directory per i database degli utenti esista
os.makedirs(app.config['USER_DB_DIR'], exist_ok=True)

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
                
                # Imposta la sessione
                session['user_id'] = user['id']
                session['username'] = username
                
                # Gestisci email e avatar (sqlite3.Row non ha il metodo get())
                if 'email' in user.keys() and user['email']:
                    session['user_email'] = user['email']
                else:
                    session['user_email'] = ''
                
                # Se l'utente ha un avatar, salvalo nella sessione
                if 'avatar' in user.keys() and user['avatar']:
                    session['user_avatar'] = user['avatar']
                    
                # Carica i crediti API nella sessione
                api_credits = get_api_credits(username)
                session['api_credits'] = api_credits
                
                # Se è una richiesta AJAX, restituisci un JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': True, 'redirect': url_for('profile')})
                
                # Altrimenti reindirizza alla pagina del profilo
                return redirect(url_for('profile'))
            else:
                error = message
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
    
    conn = get_user_db(username)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_data WHERE id = ?", (data_id,))
    data = cursor.fetchone()
    conn.close()
    
    if not data:
        return redirect(url_for('profile', error='Dati non trovati'))
    
    return render_template('view_data.html', data=data)

@app.route('/delete_user_data/<int:data_id>')
@login_required
def delete_user_data(data_id):
    username = session.get('username')
    
    try:
        conn = get_user_db(username)
        cursor = conn.cursor()
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

@app.route('/generazione-immagini')
def generazione_immagini():
    return render_template('generazione_immagini.html')

@app.route('/class_img_2')
def class_img_2():
    return render_template('class_img_2.html')

@app.route('/upload-regression', methods=['POST'])
def upload_regression():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400

    try:
        df = pd.read_csv(file)
        if len(df.columns) != 2:
            return jsonify({'error': 'Il file deve contenere esattamente due colonne'}), 400

        X = df.iloc[:, 0].values.reshape(-1, 1)
        y = df.iloc[:, 1].values

        model = LinearRegression()
        model.fit(X, y)

        # Converti i dati di training in una lista di coppie [x, y]
        training_data = [[float(x), float(y)] for x, y in zip(X.flatten(), y)]

        return jsonify({
            'success': True,
            'columns': df.columns.tolist(),
            'coefficiente': float(model.coef_[0]),
            'intercetta': float(model.intercept_),
            'training_data': training_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-regression', methods=['POST'])
def predict_regression():
    try:
        data = request.json
        value = float(data['value'])
        coef = float(data['coefficiente'])
        intercept = float(data['intercetta'])
        
        prediction = coef * value + intercept
        return jsonify({'prediction': prediction})
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
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
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

# Configurazione per il deployment
if __name__ == '__main__':
    # In ambiente di sviluppo
    app.run(debug=True)
