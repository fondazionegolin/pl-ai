from flask import Blueprint, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from werkzeug.utils import secure_filename
import os
from routes.auth import login_required
import tensorflow as tf

ml = Blueprint('ml', __name__)

# Variabili globali per i modelli
model = None
le = None

@ml.route('/regressione')
@login_required
def regressione():
    return render_template('ml/regressione.html')

@ml.route('/classificazione')
@login_required
def classificazione():
    return render_template('classificazione.html')

UPLOAD_FOLDER = 'uploads/csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@ml.route('/upload_regression', methods=['POST'])
@login_required
def upload_regression():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # Salva il file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Leggi il CSV e ottieni le informazioni
            df = pd.read_csv(filepath)
            columns = df.columns.tolist()
            preview = df.head(5).to_dict('records')
            
            return jsonify({
                'message': 'File caricato con successo',
                'filename': filename,
                'columns': columns,
                'preview': preview
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': 'File non valido'}), 400

@ml.route('/save_regression_changes', methods=['POST'])
@login_required
def save_regression_changes():
    try:
        data = request.json
        filename = data.get('filename')
        changes = data.get('changes')
        
        if not filename or not changes:
            return jsonify({'error': 'Dati mancanti'}), 400
            
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        df = pd.read_csv(filepath)
        
        # Applica le modifiche
        for change in changes:
            row = change['row']
            col = change['column']
            value = change['value']
            df.iloc[row, df.columns.get_loc(col)] = value
            
        # Salva il file modificato
        df.to_csv(filepath, index=False)
        
        return jsonify({'message': 'Modifiche salvate con successo'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@ml.route('/train_regression', methods=['POST'])
@login_required
def train_regression():
    global model
    
    try:
        data = request.json
        target = data.get('target')
        features = data.get('features')
        
        # Verifica se i dati sono stati inviati direttamente
        raw_data = data.get('data')
        
        # Se i dati non sono stati inviati direttamente, usa il nome del file
        if raw_data is None:
            filename = data.get('filename')
            if not filename:
                return jsonify({
                    'error': 'Parametri mancanti: è richiesto filename o data. ' +
                             'Suggerimento: prova a utilizzare uno dei dataset predefiniti ' +
                             'o ricarica il tuo file.'
                }), 400
            
            # Prima cerchiamo in uploads/csv
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
            
            # Se non esiste in uploads/csv, prova in static/csv/regressione
            if not os.path.exists(filepath) and 'Esempio' in filename:
                static_path = os.path.join('static', 'csv', 'regressione', secure_filename(filename))
                if os.path.exists(static_path):
                    filepath = static_path
                    print(f"Usando file da static: {filepath}")
            
            # Se dopo i tentativi ancora non esiste, ritorna un errore
            if not os.path.exists(filepath):
                available_paths = [
                    os.path.join(UPLOAD_FOLDER, secure_filename(filename)),
                    os.path.join('static', 'csv', 'regressione', secure_filename(filename))
                ]
                return jsonify({
                    'error': f'File non trovato: {filename}. ' +
                             f'Percorsi cercati: {", ".join(available_paths)}. ' +
                             'Suggerimento: utilizza uno dei dataset predefiniti o ricarica il tuo file.'
                }), 400
            
            try:
                df = pd.read_csv(filepath)
                print(f"File caricato con successo: {filepath}, shape: {df.shape}")
            except Exception as e:
                return jsonify({
                    'error': f'Errore nella lettura del file CSV: {str(e)}. ' +
                             'Suggerimento: verifica che il file sia nel formato CSV corretto.'
                }), 400
        else:
            # Crea un DataFrame dai dati ricevuti
            try:
                df_data = []
                headers = raw_data.get('headers', [])
                
                for row in raw_data.get('data', []):
                    df_data.append(row)
                
                if not df_data:
                    return jsonify({
                        'error': 'Nessun dato ricevuto. ' +
                                 'Suggerimento: riprova a caricare il dataset o utilizza uno dei dataset predefiniti.'
                    }), 400
                
                df = pd.DataFrame(df_data)
                print(f"Dati ricevuti direttamente, shape: {df.shape}")
            except Exception as e:
                return jsonify({
                    'error': f'Errore nella conversione dei dati: {str(e)}. ' +
                             'Suggerimento: ricarica il file e assicurati che sia in formato CSV valido.'
                }), 400
        
        if not all([target, features]) or len(features) == 0:
            return jsonify({
                'error': 'Parametri mancanti: target e features sono richiesti. ' +
                         'Suggerimento: seleziona almeno una colonna target e una feature.'
            }), 400
        
        # Prepara i dati
        X = df[features]
        y = df[target]
        
        # Dividi i dati
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Addestra il modello
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Valuta il modello
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        return jsonify({
            'message': 'Training completato',
            'train_score': train_score,
            'test_score': test_score,
            'features': features
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Errore dettagliato: {error_trace}")
        return jsonify({
            'error': str(e) + '. Suggerimento: riprova con un altro dataset o contatta l\'assistenza.'
        }), 400

@ml.route('/predict_regression', methods=['POST'])
@login_required
def predict_regression():
    global model
    
    if model is None:
        return jsonify({'error': 'Nessun modello addestrato'}), 400
        
    try:
        # Ottieni i valori delle features
        features = request.json.get('features', {})
        
        # Crea un DataFrame con i valori delle features
        input_data = pd.DataFrame([features])
        
        # Fai la predizione
        prediction = model.predict(input_data)[0]
        
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@ml.route('/upload_classification', methods=['POST'])
@login_required
def upload_classification():
    global model, le
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400
        
    if file:
        try:
            # Leggi il file CSV
            df = pd.read_csv(file)
            
            # Ottieni le features e il target
            features = request.form.getlist('features[]')
            target = request.form.get('target')
            
            if not features or not target:
                return jsonify({'error': 'Seleziona almeno una feature e un target'}), 400
            
            # Prepara i dati
            X = df[features]
            y = df[target]
            
            # Codifica le etichette
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Dividi i dati in training e test set
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            # Addestra il modello
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Valuta il modello
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Ottieni le classi uniche
            classes = le.classes_.tolist()
            
            return jsonify({
                'message': 'Modello addestrato con successo',
                'train_score': train_score,
                'test_score': test_score,
                'features': features,
                'classes': classes
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@ml.route('/predict_classification', methods=['POST'])
@login_required
def predict_classification():
    global model, le
    
    if model is None or le is None:
        return jsonify({'error': 'Nessun modello addestrato'}), 400
        
    try:
        # Ottieni i valori delle features
        features = request.json.get('features', {})
        
        # Crea un DataFrame con i valori delle features
        input_data = pd.DataFrame([features])
        
        # Fai la predizione
        prediction_encoded = model.predict(input_data)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        
        # Ottieni le probabilità per ogni classe
        probabilities = model.predict_proba(input_data)[0]
        prob_dict = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
        
        return jsonify({
            'prediction': prediction,
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
