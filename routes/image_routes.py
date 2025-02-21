from flask import Blueprint, request, jsonify, current_app, render_template
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
from routes.auth import login_required
from openai import OpenAI
import base64
import requests

image = Blueprint('image', __name__)

# Variabili globali
image_model = None
class_names = []
IMG_SIZE = 224

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

@image.route('/classificazione-immagini')
@login_required
def classificazione_immagini():
    return render_template('classificazione-immagini.html')

@image.route('/generazione-immagini')
@login_required
def generazione_immagini():
    return render_template('generazione-immagini.html')

@image.route('/train_image_classifier', methods=['POST'])
@login_required
def train_image_classifier():
    global image_model, class_names
    
    if 'dataset' not in request.files:
        return jsonify({'error': 'Nessun dataset caricato'}), 400
    
    try:
        # Crea cartelle temporanee per il dataset
        base_dir = 'temp_dataset'
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'validation')
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Estrai e organizza il dataset
        dataset_zip = request.files['dataset']
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # Crea i generatori di dati
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=32,
            class_mode='categorical'
        )
        
        # Salva i nomi delle classi
        class_names = list(train_generator.class_indices.keys())
        
        # Crea il modello
        image_model = Sequential([
            Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])
        
        # Compila il modello
        image_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Addestra il modello
        history = image_model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            callbacks=[CustomCallback()]
        )
        
        # Pulisci le cartelle temporanee
        import shutil
        shutil.rmtree(base_dir)
        
        return jsonify({
            'message': 'Modello addestrato con successo',
            'classes': class_names,
            'history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@image.route('/predict_image', methods=['POST'])
@login_required
def predict_image():
    global image_model, class_names
    
    if image_model is None:
        return jsonify({'error': 'Nessun modello addestrato'}), 400
        
    if 'image' not in request.files:
        return jsonify({'error': 'Nessuna immagine caricata'}), 400
        
    try:
        file = request.files['image']
        
        # Leggi e preprocessa l'immagine
        img = Image.open(file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array /= 255.
        
        # Fai la predizione
        predictions = image_model.predict(img_array)
        
        # Crea il dizionario delle predizioni
        pred_dict = {class_names[i]: float(pred) for i, pred in enumerate(predictions[0])}
        
        # Ordina le predizioni per probabilità
        sorted_predictions = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'predictions': sorted_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@image.route('/generate_image', methods=['POST'])
@login_required
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt non fornito'}), 400
            
        # Traduci e migliora il prompt se richiesto
        if data.get('translate', False):
            prompt = translate_enhance_prompt(prompt)
        
        # Genera l'immagine con DALL-E
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Ottieni l'URL dell'immagine generata
        image_url = response.data[0].url
        
        # Scarica l'immagine
        response = requests.get(image_url)
        if response.status_code == 200:
            # Converti l'immagine in base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            
            return jsonify({
                'image': image_base64,
                'prompt': prompt
            })
        else:
            return jsonify({'error': 'Errore nel download dell\'immagine'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@image.route('/translate_enhance_prompt', methods=['POST'])
@login_required
def translate_enhance_prompt():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'Prompt non fornito'}), 400
            
        # Usa GPT per tradurre e migliorare il prompt
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un esperto nella creazione di prompt per DALL-E. Il tuo compito è tradurre in inglese e migliorare i prompt forniti dagli utenti, aggiungendo dettagli artistici e tecnici per ottenere risultati migliori."},
                {"role": "user", "content": f"Traduci e migliora questo prompt per DALL-E: {prompt}"}
            ]
        )
        
        enhanced_prompt = response.choices[0].message.content
        
        return jsonify({
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
