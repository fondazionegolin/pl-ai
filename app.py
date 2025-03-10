from flask import Flask, render_template, request, jsonify, send_file
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
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime
import time

# Import dei blueprint
from routes.chatbot import chatbot
from routes.chatbot2 import chatbot2
from routes.learning import learning

# Configurazione dell'applicazione
app = Flask(__name__, static_url_path='/pl-ai/static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Registrazione dei blueprint con il prefisso
app.register_blueprint(chatbot, url_prefix='/pl-ai')
app.register_blueprint(chatbot2, url_prefix='/pl-ai')
app.register_blueprint(learning, url_prefix='/pl-ai')

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

        # Get the selected model
        model = data.get('model', 'stable-diffusion-xl-1024-v1-0')
        
        # Convert aspect ratio to dimensions based on model
        if model == 'stable-diffusion-xl-1024-v1-0':
            ratio_map = {
                '1:1': (1024, 1024),  # Square
                '3:2': (1216, 832),   # Landscape
                '2:3': (832, 1216),   # Portrait
                '16:9': (1536, 640)   # Widescreen
            }
        else:
            ratio_map = {
                '1:1': (512, 512),    # Square
                '3:2': (704, 512),    # Landscape
                '2:3': (512, 704),    # Portrait
                '16:9': (704, 384)    # Widescreen
            }
        width, height = ratio_map.get(aspect_ratio, (1024, 1024) if model == 'stable-diffusion-xl-1024-v1-0' else (512, 512))

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
        
        # Model-specific parameters
        model_params = {
            'stable-diffusion-xl-1024-v1-0': {
                'steps': 40 if high_quality else 30,
                'cfg_scale': 7,
                'style_preset': None
            },
            'stable-diffusion-512-v2-1': {
                'steps': 50 if high_quality else 35,
                'cfg_scale': 8,
                'style_preset': 'enhance'
            },
            'stable-diffusion-v1-6': {
                'steps': 60 if high_quality else 40,
                'cfg_scale': 9,
                'style_preset': 'photographic'
            }
        }
        
        params = model_params.get(model, model_params['stable-diffusion-xl-1024-v1-0'])
        
        body = {
            "width": width,
            "height": height,
            "steps": params['steps'],
            "seed": 0,
            "cfg_scale": params['cfg_scale'],
            "samples": 1,
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1
                }
            ],
            **({
                "style_preset": params['style_preset']
            } if params['style_preset'] else {}),
        }

        # Make the API request
        response = requests.post(url, headers=headers, json=body)
        
        if response.status_code != 200:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {'message': response.text}
            error_message = error_data.get('message', 'Errore sconosciuto')
            
            if 'API key' in error_message:
                error_message = 'Errore di autenticazione: chiave API non valida o mancante'
            elif 'insufficient balance' in error_message.lower():
                error_message = 'Credito insufficiente per generare l\'immagine'
            elif 'invalid parameter' in error_message.lower():
                error_message = 'Parametri di generazione non validi'
            
            raise ValueError(f'Errore durante la generazione: {error_message}')

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

    except ValueError as e:
        print(f"Validation error in generate_image: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error in generate_image: {str(e)}")
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
