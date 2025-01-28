from flask import Flask, render_template, request, jsonify, send_file
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
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime
import time

# Import dei blueprint
from routes.chatbot import chatbot
from routes.learning import learning

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Registrazione dei blueprint
app.register_blueprint(chatbot, url_prefix='')  # No prefix per mantenere gli URL come prima
app.register_blueprint(learning, url_prefix='')

# Variabili globali per i modelli
model = None
le = None
image_model = None
class_names = []
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

        return jsonify({
            'success': True,
            'columns': df.columns.tolist(),
            'coefficiente': float(model.coef_[0]),
            'intercetta': float(model.intercept_)
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
    global model, le
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400

    try:
        df = pd.read_csv(file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = RandomForestClassifier()
        model.fit(X, y_encoded)

        return jsonify({
            'success': True,
            'columns': X.columns.tolist(),
            'classes': le.classes_.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-classification', methods=['POST'])
def predict_classification():
    global model, le
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Nessun dato ricevuto'}), 400

        # Converti i dati in un formato adatto per la predizione
        features = np.array([[float(value) for value in data.values()]])
        
        # Usa il modello per fare la predizione
        prediction = model.predict(features)[0]
        predicted_class = le.inverse_transform([prediction])[0]
        
        return jsonify({'prediction': str(predicted_class)})
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

        # Convert aspect ratio to dimensions (using SDXL allowed dimensions)
        ratio_map = {
            '1:1': (1024, 1024),  # Square
            '3:2': (1216, 832),   # Landscape (closest to 3:2)
            '2:3': (832, 1216),   # Portrait (closest to 2:3)
            '16:9': (1536, 640)   # Widescreen (closest to 16:9)
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
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('STABILITY_KEY')}"
        }
        
        body = {
            "width": width,
            "height": height,
            "steps": 50 if high_quality else 30,
            "seed": 0,
            "cfg_scale": 7,
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
        
        if response.status_code != 200:
            raise ValueError(f'Errore API: {response.text}')

        # Process the response
        data = response.json()
        
        if "artifacts" not in data or len(data["artifacts"]) == 0:
            raise ValueError('Nessuna immagine generata')
            
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

    except Exception as e:
        print(f"Error in generate_image: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=True)
