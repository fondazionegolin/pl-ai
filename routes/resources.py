from flask import Blueprint, render_template, request, jsonify, send_file, redirect, url_for, session, flash, current_app
from werkzeug.utils import secure_filename
import os
import sqlite3
import uuid
import datetime
import pandas as pd
import io
import mimetypes
from PIL import Image
from routes.db import login_required

resources = Blueprint('resources', __name__)

def init_db():
    """Inizializza il database per le risorse"""
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resources
                 (id TEXT PRIMARY KEY,
                  user_id INTEGER,
                  name TEXT,
                  original_name TEXT,
                  path TEXT,
                  type TEXT,
                  size INTEGER,
                  date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Inizializza il database all'avvio
init_db()

def get_file_extension(filename):
    """Ottiene l'estensione di un file"""
    return os.path.splitext(filename)[1].lower()

def allowed_file(filename):
    """Controlla se il file ha un'estensione consentita"""
    # Consenti tutti i tipi di file per ora
    return True

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

@resources.route('/risorse')
@login_required
def risorse_page():
    """Pagina delle risorse"""
    return render_template('risorse.html')

@resources.route('/upload-resource', methods=['POST'])
@login_required
def upload_resource():
    """Carica un file come risorsa"""
    # Verifica che sia una richiesta AJAX
    if request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return jsonify({'success': False, 'error': 'Richiesta non valida'})
        
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'Nessun file inviato'})
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        if file and allowed_file(file.filename):
            # Genera un ID univoco per il file
            file_id = str(uuid.uuid4())
            
            # Ottieni l'estensione del file
            extension = get_file_extension(file.filename)
            
            # Crea un nome di file sicuro
            filename = secure_filename(file.filename)
            
            # Salva il file con un nome univoco
            file_path = os.path.join('static', 'uploads', 'resources', f"{file_id}{extension}")
            file.save(file_path)
            
            # Ottieni il tipo MIME del file
            file_type, _ = mimetypes.guess_type(file_path)
            if file_type is None:
                file_type = 'application/octet-stream'
            
            # Ottieni la dimensione del file
            file_size = os.path.getsize(file_path)
            
            # Crea una miniatura se è un'immagine
            thumbnail_path = None
            if file_type.startswith('image/'):
                thumbnail_path = os.path.join('static', 'uploads', 'resources', f"{file_id}_thumb{extension}")
                create_thumbnail(file_path, thumbnail_path)
            
            # Salva le informazioni nel database
            conn = sqlite3.connect('database.sqlite')
            c = conn.cursor()
            c.execute('''INSERT INTO resources (id, user_id, name, original_name, path, type, size, date)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (file_id, session['user_id'], f"{file_id}{extension}", filename, file_path, file_type, file_size, datetime.datetime.now()))
            conn.commit()
            conn.close()
            
            uploaded_files.append({
                'id': file_id,
                'name': filename,
                'path': file_path,
                'type': file_type,
                'size': file_size,
                'thumbnail': thumbnail_path
            })
    
    return jsonify({'success': True, 'files': uploaded_files})

@resources.route('/get-resources')
@login_required
def get_resources():
    """Ottiene la lista delle risorse dell'utente"""
    # Verifica che sia una richiesta AJAX
    if request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return jsonify({'success': False, 'error': 'Richiesta non valida'})
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('''SELECT id, name, original_name, path, type, size, date
                 FROM resources
                 WHERE user_id = ?
                 ORDER BY date DESC''', (session['user_id'],))
    resources_data = c.fetchall()
    conn.close()
    
    files = []
    for resource in resources_data:
        file_id, name, original_name, path, file_type, size, date = resource
        
        # Controlla se esiste una miniatura per le immagini
        thumbnail = None
        if file_type and file_type.startswith('image/'):
            extension = get_file_extension(name)
            thumbnail_path = os.path.join('static', 'uploads', 'resources', f"{file_id}_thumb{extension}")
            if os.path.exists(thumbnail_path):
                thumbnail = f"/static/uploads/resources/{file_id}_thumb{extension}"
        
        files.append({
            'id': file_id,
            'name': original_name,
            'path': f"/static/uploads/resources/{name}",
            'type': file_type,
            'size': size,
            'date': date,
            'thumbnail': thumbnail
        })
    
    return jsonify({'success': True, 'files': files})

@resources.route('/api/files')
@login_required
def get_files_for_rag():
    """Ottiene la lista dei file disponibili per la modalità RAG"""
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('''SELECT id, original_name, type, size, date
                 FROM resources
                 WHERE user_id = ?
                 ORDER BY date DESC''', (session['user_id'],))
    resources_data = c.fetchall()
    conn.close()
    
    files = []
    for resource in resources_data:
        file_id, original_name, file_type, size, date = resource
        
        # Filtra i tipi di file supportati per RAG
        # Supportiamo PDF, immagini, CSV e file di testo
        if file_type and (file_type.startswith('image/') or 
                         file_type == 'application/pdf' or 
                         file_type == 'text/csv' or 
                         file_type == 'text/plain'):
            files.append({
                'id': file_id,
                'name': original_name,
                'type': file_type,
                'size': size,
                'date': date
            })
    
    return jsonify({'files': files})

@resources.route('/download-resource/<file_id>')
@login_required
def download_resource(file_id):
    """Scarica una risorsa"""
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('''SELECT path, original_name, type
                 FROM resources
                 WHERE id = ? AND user_id = ?''', (file_id, session['user_id']))
    resource = c.fetchone()
    conn.close()
    
    if not resource:
        flash('Risorsa non trovata', 'error')
        return redirect(url_for('resources.risorse_page'))
    
    file_path, original_name, file_type = resource
    
    return send_file(file_path, as_attachment=True, download_name=original_name, mimetype=file_type)

@resources.route('/delete-resource/<file_id>', methods=['DELETE'])
@login_required
def delete_resource(file_id):
    """Elimina una risorsa"""
    # Verifica che sia una richiesta AJAX
    if request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return jsonify({'success': False, 'error': 'Richiesta non valida'})
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    
    # Ottieni le informazioni sul file
    c.execute('''SELECT path, name, type
                 FROM resources
                 WHERE id = ? AND user_id = ?''', (file_id, session['user_id']))
    resource = c.fetchone()
    
    if not resource:
        conn.close()
        return jsonify({'success': False, 'error': 'Risorsa non trovata'})
    
    file_path, name, file_type = resource
    
    # Elimina il file dal database
    c.execute('''DELETE FROM resources
                 WHERE id = ? AND user_id = ?''', (file_id, session['user_id']))
    conn.commit()
    conn.close()
    
    # Elimina il file dal filesystem
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Elimina anche la miniatura se esiste
        if file_type and file_type.startswith('image/'):
            extension = get_file_extension(name)
            thumbnail_path = os.path.join('static', 'uploads', 'resources', f"{file_id}_thumb{extension}")
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@resources.route('/preview-csv/<file_id>')
@login_required
def preview_csv(file_id):
    """Anteprima di un file CSV"""
    # Verifica che sia una richiesta AJAX
    if request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return jsonify({'success': False, 'error': 'Richiesta non valida'})
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('''SELECT path
                 FROM resources
                 WHERE id = ? AND user_id = ? AND type = 'text/csv' ''', (file_id, session['user_id']))
    resource = c.fetchone()
    conn.close()
    
    if not resource:
        return jsonify({'success': False, 'error': 'File CSV non trovato'})
    
    file_path = resource[0]
    
    try:
        # Prova a leggere il file CSV con diversi encoding
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        df = None
        error_msg = ""
        
        for encoding in encodings:
            try:
                # Prova a leggere il file con l'encoding corrente
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                break  # Se ha funzionato, esci dal ciclo
            except Exception as e:
                error_msg = str(e)
                continue
        
        if df is None:
            # Se nessun encoding ha funzionato
            return jsonify({'success': False, 'error': f'Impossibile leggere il file CSV: {error_msg}'})
        
        # Limita il numero di righe per l'anteprima
        preview_rows = 100
        if len(df) > preview_rows:
            df = df.head(preview_rows)
        
        # Converti in HTML con stili migliorati
        html = df.to_html(classes='min-w-full divide-y divide-gray-200 table-auto border-collapse', index=False)
        html = html.replace('<table', '<table style="width:100%; border-collapse: collapse;"')
        html = html.replace('<th', '<th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2; text-align: left;"')
        html = html.replace('<td', '<td style="border: 1px solid #ddd; padding: 8px;"')
        
        return jsonify({'success': True, 'html': html})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Errore durante l\'anteprima del CSV: {str(e)}'})
