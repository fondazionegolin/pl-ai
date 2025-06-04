import os
import json
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import openai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import uuid
import pickle

# Crea il blueprint
machine_learning_bp = Blueprint('machine_learning', __name__)

# Configurazione
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_FOLDER = 'models/ml_models'

# Assicurati che le cartelle esistano
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Funzione per verificare l'estensione del file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route per la pagina principale
@machine_learning_bp.route('/machine_learning')
def machine_learning():
    return render_template('machine_learning.html')

# Route per l'analisi del dataset
@machine_learning_bp.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Nessun file caricato'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nessun file selezionato'})
    
    if file and allowed_file(file.filename):
        # Salva il file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Genera un ID univoco per questa sessione di analisi
        session_id = str(uuid.uuid4())
        session['ml_session_id'] = session_id
        session['dataset_path'] = filepath
        
        try:
            # Leggi il CSV
            df = pd.read_csv(filepath)
            
            # Salva una copia del dataframe
            session['dataset_columns'] = df.columns.tolist()
            
            # Prepara i dati per l'analisi OpenAI
            sample_data = df.head(5).to_string()
            column_info = {}
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    column_info[col] = {
                        'type': 'numeric',
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'missing': int(df[col].isna().sum())
                    }
                else:
                    column_info[col] = {
                        'type': 'categorical',
                        'unique_values': df[col].nunique(),
                        'top_values': df[col].value_counts().head(5).to_dict(),
                        'missing': int(df[col].isna().sum())
                    }
            
            # Analisi con OpenAI
            description = analyze_with_openai(df, sample_data, column_info)
            
            # Salva il dataframe per uso futuro
            df_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_dataset.pkl")
            df.to_pickle(df_path)
            session['df_path'] = df_path
            
            return jsonify({
                'success': True,
                'description': description,
                'columns': df.columns.tolist(),
                'data': {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Formato file non supportato'})

# Funzione per analizzare il dataset con OpenAI
def analyze_with_openai(df, sample_data, column_info):
    try:
        # Prepara il prompt per OpenAI
        prompt = f"""
        Analizza il seguente dataset:
        
        Dimensioni: {df.shape[0]} righe x {df.shape[1]} colonne
        
        Prime 5 righe:
        {sample_data}
        
        Informazioni sulle colonne:
        {json.dumps(column_info, indent=2)}
        
        Fornisci una descrizione dettagliata del dataset, includendo:
        1. Una panoramica generale di cosa contiene il dataset
        2. Il tipo di dati presenti in ogni colonna
        3. Eventuali correlazioni evidenti tra le colonne
        4. Potenziali problemi come valori mancanti o anomali
        5. Suggerimenti su quali colonne potrebbero essere buoni target per modelli di machine learning
        
        Formatta la risposta in HTML con paragrafi, elenchi e sezioni ben organizzate.
        """
        
        # Chiamata all'API di OpenAI (versione 1.0.0+)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un esperto di analisi dati e machine learning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        # Estrai la descrizione dalla risposta
        description = response.choices[0].message.content
        return description
    
    except Exception as e:
        print(f"Errore nell'analisi OpenAI: {str(e)}")
        return f"<p>Non è stato possibile generare un'analisi automatica. Errore: {str(e)}</p>"

# Route per l'analisi della colonna target
@machine_learning_bp.route('/analyze_target', methods=['POST'])
def analyze_target():
    data = request.json
    target_column = data.get('target_column')
    
    if not target_column or 'df_path' not in session:
        return jsonify({'success': False, 'error': 'Dati mancanti o sessione scaduta'})
    
    try:
        # Carica il dataframe
        df = pd.read_pickle(session['df_path'])
        
        # Determina il tipo di problema
        is_categorical = not pd.api.types.is_numeric_dtype(df[target_column]) or df[target_column].nunique() < 10
        problem_type = 'classification' if is_categorical else 'regression'
        
        # Salva il tipo di problema e la colonna target nella sessione
        session['problem_type'] = problem_type
        session['target_column'] = target_column
        
        # Prepara le informazioni per l'analisi OpenAI
        target_info = {}
        if problem_type == 'classification':
            target_info = {
                'type': 'categorical',
                'unique_values': df[target_column].nunique(),
                'value_counts': df[target_column].value_counts().to_dict()
            }
        else:
            target_info = {
                'type': 'numeric',
                'min': float(df[target_column].min()),
                'max': float(df[target_column].max()),
                'mean': float(df[target_column].mean()),
                'std': float(df[target_column].std())
            }
        
        # Ottieni la raccomandazione dell'algoritmo
        recommendation = recommend_algorithm(df, target_column, problem_type, target_info)
        
        return jsonify({
            'success': True,
            'problem_type': problem_type,
            'analysis': recommendation,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Funzione per raccomandare un algoritmo
def recommend_algorithm(df, target_column, problem_type, target_info):
    try:
        # Prepara il prompt per OpenAI
        feature_columns = [col for col in df.columns if col != target_column]
        numeric_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_features = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        prompt = f"""
        Analizza il seguente problema di machine learning:
        
        Tipo di problema: {problem_type}
        Colonna target: {target_column}
        
        Informazioni sulla colonna target:
        {json.dumps(target_info, indent=2)}
        
        Feature numeriche disponibili: {len(numeric_features)}
        Feature categoriche disponibili: {len(categorical_features)}
        
        Dimensioni del dataset: {df.shape[0]} righe x {df.shape[1]} colonne
        
        Fornisci SOLO una breve risposta (massimo 3 frasi) che indichi se è meglio usare:
        1. Regressione lineare
        2. Regressione non lineare
        3. Classificazione
        
        Spiega brevemente il perché della tua scelta. Formatta la risposta in un breve paragrafo HTML.
        """
        
        # Chiamata all'API di OpenAI (versione 1.0.0+)
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sei un esperto di machine learning che fornisce raccomandazioni di algoritmi."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        # Estrai la raccomandazione dalla risposta
        recommendation = response.choices[0].message.content
        return recommendation
    
    except Exception as e:
        print(f"Errore nella raccomandazione dell'algoritmo: {str(e)}")
        return f"<p>Non è stato possibile generare una raccomandazione. Errore: {str(e)}</p>"

# Route per il training del modello
@machine_learning_bp.route('/train_model', methods=['POST'])
def train_model():
    if 'df_path' not in session:
        return jsonify({'success': False, 'error': 'Sessione scaduta'})
    
    # Ottieni i parametri dalla richiesta o usa valori predefiniti
    target_column = request.form.get('target_column')
    problem_type = request.form.get('problem_type')
    
    # Carica il dataframe
    try:
        df = pd.read_pickle(session['df_path'])
    except Exception as e:
        return jsonify({'success': False, 'error': f'Errore nel caricamento del dataset: {str(e)}'})
    
    # Se la colonna target non è specificata, usa l'ultima colonna
    if not target_column and len(df.columns) > 0:
        target_column = df.columns[-1]
    elif not target_column:
        return jsonify({'success': False, 'error': 'Nessuna colonna disponibile nel dataset'})
    
    # Mappa i tipi di problema specifici ai tipi generici
    problem_type_mapping = {
        'regressione_lineare': 'regression',
        'regressione_non_lineare': 'regression',
        'classificazione': 'classification'
    }
    
    # Usa il tipo di problema mappato o il valore predefinito
    mapped_problem_type = problem_type_mapping.get(problem_type, 'regression')
    
    try:
        # Prepara i dati
        X, y, feature_names, preprocessor = prepare_data(df, target_column, mapped_problem_type)
        
        # Dividi i dati in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scegli il modello in base al tipo di problema specifico
        if problem_type == 'regressione_lineare':
            # Usa una regressione lineare semplice
            if mapped_problem_type == 'regression':
                model = LinearRegression()
            else:  # classificazione
                model = LogisticRegression(max_iter=1000)
        elif problem_type == 'regressione_non_lineare':
            # Usa un modello non lineare
            if mapped_problem_type == 'regression':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:  # classificazione
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # classificazione o default
            if mapped_problem_type == 'regression':
                model = GradientBoostingRegressor(random_state=42)
            else:  # classificazione
                model = GradientBoostingClassifier(random_state=42)
        
        # Addestra il modello
        model.fit(X_train, y_train)
        
        # Valuta il modello
        y_pred = model.predict(X_test)
        
        # Calcola le metriche
        metrics = {}
        if mapped_problem_type == 'regression':
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
            metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])
        else:  # classificazione
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Salva il modello
        model_path = os.path.join(MODEL_FOLDER, f"{session['ml_session_id']}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'problem_type': mapped_problem_type,  # Salva il tipo di problema mappato
                'original_problem_type': problem_type,  # Salva anche il tipo originale
                'target_column': target_column
            }, f)
        
        session['model_path'] = model_path
        
        # Prepara i dati per i grafici
        plot_data = prepare_plot_data(df, X, y, model, preprocessor, feature_names, target_column, mapped_problem_type)
        
        # Prepara le informazioni sulle feature per la predizione
        feature_info = {}
        for i, feature in enumerate(feature_names):
            if pd.api.types.is_numeric_dtype(df[feature]):
                feature_info[feature] = {
                    'type': 'numeric',
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'default': float(df[feature].mean())
                }
            else:
                feature_info[feature] = {
                    'type': 'categorical',
                    'values': df[feature].unique().tolist()
                }
        
        # Prepara i dati della curva di fit per regressione
        fit_curve = None
        if mapped_problem_type == 'regression' and len(feature_names) == 1:
            fit_curve = prepare_fit_curve(df, feature_names[0], target_column, model, preprocessor)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'plot_data': plot_data,
            'fit_curve': fit_curve,
            'feature_info': feature_info,
            'model_info': {
                'type': problem_type,  # Tipo di problema originale per l'interfaccia utente
                'mapped_type': mapped_problem_type,  # Tipo di problema mappato per il backend
                'features': feature_names,
                'target': target_column
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Funzione per preparare i dati
def prepare_data(df, target_column, problem_type):
    # Separa feature e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Salva i nomi delle feature originali
    feature_names = X.columns.tolist()
    
    # Preprocessamento
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Crea un preprocessore
    preprocessor = {}
    
    # Gestisci le feature numeriche
    if numeric_features:
        scaler = StandardScaler()
        X_numeric = X[numeric_features].copy()
        X_numeric = X_numeric.fillna(X_numeric.mean())
        X_numeric = scaler.fit_transform(X_numeric)
        preprocessor['scaler'] = scaler
        preprocessor['numeric_features'] = numeric_features
    else:
        X_numeric = np.array([]).reshape(X.shape[0], 0)
    
    # Gestisci le feature categoriche
    if categorical_features:
        encoders = {}
        X_categorical_encoded = np.zeros((X.shape[0], 0))
        
        for feature in categorical_features:
            encoder = LabelEncoder()
            # Gestisci i valori mancanti
            X[feature] = X[feature].fillna('missing')
            encoded = encoder.fit_transform(X[feature])
            encoded = encoded.reshape(-1, 1)
            X_categorical_encoded = np.hstack((X_categorical_encoded, encoded))
            encoders[feature] = encoder
        
        preprocessor['encoders'] = encoders
        preprocessor['categorical_features'] = categorical_features
    else:
        X_categorical_encoded = np.array([]).reshape(X.shape[0], 0)
    
    # Combina le feature preprocessate
    X_processed = np.hstack((X_numeric, X_categorical_encoded))
    
    # Gestisci il target per la classificazione
    if problem_type == 'classification':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        preprocessor['target_encoder'] = target_encoder
    
    return X_processed, y, feature_names, preprocessor

# Funzione per addestrare e valutare il modello
def train_and_evaluate(X_train, X_test, y_train, y_test, problem_type):
    if problem_type == 'regression':
        # Per dataset piccoli, usa LinearRegression
        if X_train.shape[0] < 1000:
            model = LinearRegression()
        # Per dataset più grandi, usa RandomForestRegressor
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Addestra il modello
        model.fit(X_train, y_train)
        
        # Valuta il modello
        y_pred = model.predict(X_test)
        
        # Calcola le metriche
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    else:  # classification
        # Per dataset piccoli, usa LogisticRegression
        if X_train.shape[0] < 1000:
            model = LogisticRegression(max_iter=1000, random_state=42)
        # Per dataset più grandi, usa RandomForestClassifier
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Addestra il modello
        model.fit(X_train, y_train)
        
        # Valuta il modello
        y_pred = model.predict(X_test)
        
        # Calcola le metriche
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per classificazione binaria
        if len(np.unique(y_test)) == 2:
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
        else:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return model, metrics

# Funzione per preparare i dati per i grafici
def prepare_plot_data(df, X, y, model, preprocessor, feature_names, target_column, problem_type):
    # Per semplicità, usiamo solo le prime due feature per il grafico di dispersione
    # Se c'è una sola feature, la usiamo due volte
    if len(feature_names) == 1:
        x_feature = feature_names[0]
        x_label = x_feature
        y_label = target_column
        
        # Prepara i punti
        points = []
        for i in range(min(1000, len(df))):  # Limita a 1000 punti per performance
            points.append({
                'x': float(df[x_feature].iloc[i]),
                'y': float(df[target_column].iloc[i]) if problem_type == 'regression' else str(df[target_column].iloc[i])
            })
    else:
        # Scegli le due feature più importanti
        if hasattr(model, 'feature_importances_'):
            # Per modelli con feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [feature_names[i] for i in indices[:2]]
        else:
            # Altrimenti, prendi le prime due
            top_features = feature_names[:2]
        
        x_feature, y_feature = top_features
        x_label = x_feature
        y_label = y_feature
        
        # Prepara i punti
        points = []
        for i in range(min(1000, len(df))):  # Limita a 1000 punti per performance
            points.append({
                'x': float(df[x_feature].iloc[i]),
                'y': float(df[y_feature].iloc[i])
            })
    
    return {
        'points': points,
        'x_label': x_label,
        'y_label': y_label
    }

# Funzione per preparare i dati della curva di fit per regressione
def prepare_fit_curve(df, feature, target, model, preprocessor):
    # Questa funzione è per regressione con una sola feature
    x_values = np.linspace(df[feature].min(), df[feature].max(), 100)
    
    # Prepara i dati per la predizione
    X_pred = np.zeros((len(x_values), 1))
    X_pred[:, 0] = x_values
    
    # Applica il preprocessore
    if 'scaler' in preprocessor and feature in preprocessor['numeric_features']:
        idx = preprocessor['numeric_features'].index(feature)
        X_pred = preprocessor['scaler'].transform(X_pred)
    
    # Predici i valori y
    y_pred = model.predict(X_pred)
    
    # Prepara i punti per il grafico
    points = []
    for i in range(min(1000, len(df))):
        points.append({
            'x': float(df[feature].iloc[i]),
            'y': float(df[target].iloc[i])
        })
    
    # Prepara i punti della curva
    curve = []
    for i in range(len(x_values)):
        curve.append({
            'x': float(x_values[i]),
            'y': float(y_pred[i])
        })
    
    return {
        'points': points,
        'curve': curve,
        'x_label': feature,
        'y_label': target
    }

# Route per la predizione
@machine_learning_bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get('features')
    
    if not features or 'model_path' not in session:
        return jsonify({'success': False, 'error': 'Dati mancanti o sessione scaduta'})
    
    try:
        # Carica il modello
        with open(session['model_path'], 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        feature_names = model_data['feature_names']
        problem_type = model_data['problem_type']
        
        # Prepara i dati di input
        input_data = {}
        for feature in feature_names:
            if feature in features:
                input_data[feature] = features[feature]
            else:
                return jsonify({'success': False, 'error': f'Feature mancante: {feature}'})
        
        # Converti in DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocessa i dati
        X_input = preprocess_input(input_df, preprocessor, feature_names)
        
        # Fai la predizione
        prediction = model.predict(X_input)[0]
        
        # Decodifica la predizione per classificazione
        if problem_type == 'classification' and 'target_encoder' in preprocessor:
            prediction = preprocessor['target_encoder'].inverse_transform([prediction])[0]
        
        # Formatta la predizione
        if isinstance(prediction, (np.float64, np.float32, float)):
            prediction = round(float(prediction), 4)
        else:
            prediction = str(prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Funzione per preprocessare i dati di input
def preprocess_input(input_df, preprocessor, feature_names):
    # Gestisci le feature numeriche
    if 'numeric_features' in preprocessor and preprocessor['numeric_features']:
        numeric_features = preprocessor['numeric_features']
        X_numeric = input_df[numeric_features].copy()
        X_numeric = X_numeric.fillna(0)  # Sostituisci con la media se disponibile
        X_numeric = preprocessor['scaler'].transform(X_numeric)
    else:
        X_numeric = np.array([]).reshape(input_df.shape[0], 0)
    
    # Gestisci le feature categoriche
    if 'categorical_features' in preprocessor and preprocessor['categorical_features']:
        encoders = preprocessor['encoders']
        categorical_features = preprocessor['categorical_features']
        X_categorical_encoded = np.zeros((input_df.shape[0], 0))
        
        for feature in categorical_features:
            encoder = encoders[feature]
            # Gestisci i valori mancanti
            input_df[feature] = input_df[feature].fillna('missing')
            try:
                encoded = encoder.transform(input_df[feature])
            except:
                # Se il valore non è stato visto durante il training, usa il primo valore
                encoded = np.array([0])
            encoded = encoded.reshape(-1, 1)
            X_categorical_encoded = np.hstack((X_categorical_encoded, encoded))
    else:
        X_categorical_encoded = np.array([]).reshape(input_df.shape[0], 0)
    
    # Combina le feature preprocessate
    X_processed = np.hstack((X_numeric, X_categorical_encoded))
    
    return X_processed
