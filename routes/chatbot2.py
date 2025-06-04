from flask import Blueprint, render_template, request, jsonify, session
import sqlite3
import json
import os
import base64
import tempfile
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF per l'elaborazione di PDF
import pandas as pd
import pytesseract
from PIL import Image
import tiktoken  # Per il conteggio dei token
import requests
import time

load_dotenv()

# Initialize API clients and validate API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Dictionary to track API client initialization status
api_status = {
    'openai': False,
    'anthropic': False,
    'gemini': False,
    'ollama': False
}

# Ollama API endpoint
# Leggi l'endpoint Ollama da una variabile d'ambiente o usa un valore predefinito
OLLAMA_API_ENDPOINT = os.getenv('OLLAMA_API_ENDPOINT', "http://localhost:11434")

# Last Ollama API request timestamp (for rate limiting)
ollama_last_request_time = 0
OLLAMA_RATE_LIMIT_DELAY = 1.0  # Minimum seconds between Ollama API requests

# Don't check Ollama on startup to avoid blocking the service
# We'll check it on-demand when a user tries to use it
print("Ollama API check deferred until needed")
api_status['ollama'] = False  # Will be checked on-demand

# Initialize OpenAI client
try:
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
        # Test the client with a simple request
        openai_client.models.list()
        api_status['openai'] = True
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    openai_client = None

# Initialize Anthropic client
try:
    if anthropic_api_key:
        try:
            # Try the newer version of the Anthropic client
            anthropic_client = Anthropic(api_key=anthropic_api_key)
        except TypeError:
            # Fall back to older version which might have different parameters
            import pkg_resources
            anthropic_version = pkg_resources.get_distribution("anthropic").version
            print(f"Using Anthropic library version: {anthropic_version}")
            
            # For older versions (pre-0.5.0)
            if pkg_resources.parse_version(anthropic_version) < pkg_resources.parse_version("0.5.0"):
                anthropic_client = Anthropic(api_key=anthropic_api_key)
            else:
                # For newer versions that might have different parameters
                anthropic_client = Anthropic()
        
        api_status['anthropic'] = True
    else:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error initializing Anthropic client: {str(e)}")
    anthropic_client = None

# Initialize Gemini
try:
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        api_status['gemini'] = True
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error initializing Gemini client: {str(e)}")

chatbot2 = Blueprint('chatbot2', __name__)

def init_db():
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    
    # Crea la tabella delle chat con il campo user_id
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot2_chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  title TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot2_messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id INTEGER,
                  role TEXT,
                  content TEXT,
                  model TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (chat_id) REFERENCES chatbot2_chats(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot2_settings
                 (chat_id INTEGER PRIMARY KEY,
                  grade TEXT,
                  mode TEXT,
                  subject TEXT,
                  model TEXT,
                  system_prompt TEXT,
                  FOREIGN KEY (chat_id) REFERENCES chatbot2_chats(id))''')
    
    # Verifica se la colonna user_id esiste già, altrimenti la aggiunge
    try:
        # Controlla se la colonna user_id esiste
        c.execute("SELECT user_id FROM chatbot2_chats LIMIT 1")
    except sqlite3.OperationalError:
        # Se non esiste, aggiungila
        c.execute("ALTER TABLE chatbot2_chats ADD COLUMN user_id INTEGER")
    conn.commit()
    conn.close()

# Funzioni per l'elaborazione dei documenti per RAG
def extract_text_from_pdf(pdf_path):
    """Estrae il testo da un file PDF usando pdfplumber"""
    try:
        print(f"Estrazione testo da PDF: {pdf_path}")
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"PDF contiene {num_pages} pagine")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                print(f"Estratti {len(page_text)} caratteri dalla pagina {i+1}")
        
        print(f"Estrazione completata: {len(text)} caratteri totali")
        return text
    except Exception as e:
        print(f"Errore nell'estrazione del testo dal PDF: {e}")
        import traceback
        traceback.print_exc()
        return ""

def extract_text_from_image(image_path):
    """Estrae il testo da un'immagine usando OCR"""
    try:
        print(f"Estrazione testo da immagine: {image_path}")
        image = Image.open(image_path)
        print(f"Immagine caricata: dimensioni {image.width}x{image.height}, formato {image.format}")
        
        text = pytesseract.image_to_string(image, lang='ita+eng')
        print(f"Estrazione completata: {len(text)} caratteri estratti")
        return text
    except Exception as e:
        print(f"Errore nell'estrazione del testo dall'immagine: {e}")
        import traceback
        traceback.print_exc()
        return ""

def extract_text_from_csv(csv_path):
    """Estrae il testo da un file CSV"""
    try:
        print(f"Estrazione testo da CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Converti il DataFrame in una stringa formattata
        text = df.to_string()
        print(f"Estrazione completata: {len(text)} caratteri estratti")
        return text
    except Exception as e:
        print(f"Errore nell'estrazione del testo dal CSV: {e}")
        import traceback
        traceback.print_exc()
        return ""

def get_document_content(file_id):
    """Ottiene il contenuto di un documento in base al suo ID"""
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('SELECT path, type FROM resources WHERE id = ?', (file_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        print(f"File non trovato nel database: {file_id}")
        return ""
    
    file_path, file_type = result
    
    # Verifica se il percorso è già assoluto o relativo
    if not os.path.isabs(file_path):
        # Se è relativo, costruisci il percorso assoluto
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        print(f"File non trovato sul disco: {file_path}")
        return ""
    
    # Estrai il testo in base al tipo di file
    if file_type == 'application/pdf':
        return extract_text_from_pdf(file_path)
    elif file_type.startswith('image/'):
        return extract_text_from_image(file_path)
    elif file_type == 'text/csv':
        return extract_text_from_csv(file_path)
    elif file_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""

# Function to count tokens in messages
def count_tokens(messages):
    """Conta il numero di token in una lista di messaggi.
    
    Args:
        messages: Lista di dizionari con 'role' e 'content'.
        
    Returns:
        Numero di token.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = 0
        for message in messages:
            # Ogni messaggio ha un token base di 3 per modelli come GPT-3.5 e 4
            num_tokens += 3
            # Aggiungi token per il ruolo
            num_tokens += len(encoding.encode(message.get('role', '')))
            # Aggiungi token per il contenuto
            num_tokens += len(encoding.encode(message.get('content', '')))
        # Aggiungi un token base per la richiesta generale
        num_tokens += 3
        return num_tokens
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        # Fallback approssimativo se tiktoken fallisce
        total_chars = sum(len(m.get('content', '')) for m in messages)
        return total_chars // 4  # Approssimazione grossolana: ~4 caratteri per token

# Initialize database tables
init_db()

@chatbot2.route('/chatbot2')
def chat_page():
    return render_template('chatbot2.html')

@chatbot2.route('/api/chat-history')
def get_chat_history():
    # Verifica se l'utente è autenticato
    user_id = session.get('user_id')
    
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    
    if user_id:
        # Se l'utente è autenticato, mostra solo le sue chat
        c.execute('SELECT id, title, created_at FROM chatbot2_chats WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    else:
        # Se l'utente non è autenticato, mostra solo le chat senza user_id (chat anonime)
        c.execute('SELECT id, title, created_at FROM chatbot2_chats WHERE user_id IS NULL ORDER BY created_at DESC')
    
    chats = [{'id': row[0], 'title': row[1], 'created_at': row[2]} for row in c.fetchall()]
    conn.close()
    return jsonify(chats)

@chatbot2.route('/api/chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    conn = None
    try:
        # Verifica se l'utente è autenticato
        user_id = session.get('user_id')
        
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        
        # Verifica che la chat esista e appartenga all'utente corrente
        if user_id:
            c.execute('SELECT id FROM chatbot2_chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
        else:
            c.execute('SELECT id FROM chatbot2_chats WHERE id = ? AND user_id IS NULL', (chat_id,))
            
        if not c.fetchone():
            return jsonify({'success': False, 'error': 'Chat non trovata o non autorizzata'}), 404
        
        # Delete chat messages
        c.execute('DELETE FROM chatbot2_messages WHERE chat_id = ?', (chat_id,))
        
        # Delete chat settings
        c.execute('DELETE FROM chatbot2_settings WHERE chat_id = ?', (chat_id,))
        
        # Delete chat
        c.execute('DELETE FROM chatbot2_chats WHERE id = ?', (chat_id,))
        
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

@chatbot2.route('/api/chats', methods=['DELETE'])
def delete_all_chats():
    conn = None
    try:
        # Verifica se l'utente è autenticato
        user_id = session.get('user_id')
        
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        
        if user_id:
            # Ottieni tutte le chat dell'utente
            c.execute('SELECT id FROM chatbot2_chats WHERE user_id = ?', (user_id,))
            chat_ids = [row[0] for row in c.fetchall()]
            
            if chat_ids:
                # Elimina i messaggi delle chat dell'utente
                c.execute('DELETE FROM chatbot2_messages WHERE chat_id IN ({})'.format(','.join('?' * len(chat_ids))), chat_ids)
                
                # Elimina le impostazioni delle chat dell'utente
                c.execute('DELETE FROM chatbot2_settings WHERE chat_id IN ({})'.format(','.join('?' * len(chat_ids))), chat_ids)
                
                # Elimina le chat dell'utente
                c.execute('DELETE FROM chatbot2_chats WHERE user_id = ?', (user_id,))
        else:
            # Ottieni tutte le chat anonime
            c.execute('SELECT id FROM chatbot2_chats WHERE user_id IS NULL')
            chat_ids = [row[0] for row in c.fetchall()]
            
            if chat_ids:
                # Elimina i messaggi delle chat anonime
                c.execute('DELETE FROM chatbot2_messages WHERE chat_id IN ({})'.format(','.join('?' * len(chat_ids))), chat_ids)
                
                # Elimina le impostazioni delle chat anonime
                c.execute('DELETE FROM chatbot2_settings WHERE chat_id IN ({})'.format(','.join('?' * len(chat_ids))), chat_ids)
                
                # Elimina le chat anonime
                c.execute('DELETE FROM chatbot2_chats WHERE user_id IS NULL')
        
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

@chatbot2.route('/api/chat/<int:chat_id>')
def get_chat(chat_id):
    # Verifica se l'utente è autenticato
    user_id = session.get('user_id')
    
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    
    # Verifica che la chat esista e appartenga all'utente corrente
    if user_id:
        c.execute('SELECT id FROM chatbot2_chats WHERE id = ? AND user_id = ?', (chat_id, user_id))
    else:
        c.execute('SELECT id FROM chatbot2_chats WHERE id = ? AND user_id IS NULL', (chat_id,))
        
    if not c.fetchone():
        return jsonify({'success': False, 'error': 'Chat non trovata o non autorizzata'}), 404
    
    # Get chat messages
    c.execute('SELECT role, content, model FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at', (chat_id,))
    messages = [{'role': row[0], 'content': row[1], 'model': row[2]} for row in c.fetchall()]
    
    # Get chat settings
    c.execute('SELECT grade, mode, subject, model, system_prompt FROM chatbot2_settings WHERE chat_id = ?', (chat_id,))
    settings = c.fetchone()
    
    conn.close()
    
    return jsonify({
        'messages': messages,
        'settings': {
            'grade': settings[0] if settings else None,
            'mode': settings[1] if settings else None,
            'subject': settings[2] if settings else None,
            'model': settings[3] if settings else None,
            'systemPrompt': settings[4] if settings else None
        }
    })

@chatbot2.route('/api/chat', methods=['POST'])
def chat():
    conn = None
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        context = data.get('context', {})
        chat_id = data.get('chatId')
        
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        
        # Create new chat if needed
        if not chat_id:
            # Verifica se l'utente è autenticato
            user_id = session.get('user_id')
            
            if user_id:
                # Se l'utente è autenticato, associa la chat all'utente
                c.execute('INSERT INTO chatbot2_chats (user_id, title) VALUES (?, ?)', (user_id, message[:50]))
            else:
                # Se l'utente non è autenticato, crea una chat anonima
                c.execute('INSERT INTO chatbot2_chats (title) VALUES (?)', (message[:50],))
                
            chat_id = c.lastrowid
            
            # Save chat settings
            c.execute('''INSERT INTO chatbot2_settings 
                         (chat_id, grade, mode, subject, model, system_prompt)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                     (chat_id, context.get('grade'), context.get('mode'),
                      context.get('subject'), context.get('model'),
                      context.get('systemPrompt')))
        
        # Save user message
        c.execute('''INSERT INTO chatbot2_messages (chat_id, role, content, model)
                     VALUES (?, ?, ?, ?)''',
                 (chat_id, 'user', message, context.get('model')))
        
        # Get AI response based on selected model and mode
        model_name = context.get('model', 'gpt4')
        mode = context.get('mode', '')
        response = None
        system_prompt = context.get('systemPrompt', '')
        
        # Special handling for different modes
        if mode == 'interrogazione' and message == 'START_INTERROGATION':
            # Set initial title for interrogation
            c.execute('UPDATE chatbot2_chats SET title = ? WHERE id = ?',
                     (f"Interrogazione di {context.get('subject', 'materia')}", chat_id))
            conn.commit()
            response = "Iniziamo l'interrogazione. Ti farò delle domande sull'argomento specificato."
        elif mode == 'intervista':
            # For interview mode, always include the context in each message
            # to maintain character consistency
            system_prompt = context.get('systemPrompt', '')
            
            # Extract character name from system prompt
            import re
            character_match = re.search(r'Interpreta il personaggio storico: ([^.\n]+)', system_prompt)
            if character_match:
                character_name = character_match.group(1).strip()
                if message == 'START_INTERVIEW':
                    # Set initial title and welcome message for interview
                    c.execute('UPDATE chatbot2_chats SET title = ? WHERE id = ?',
                             (f"Intervista a {character_name}", chat_id))
                    conn.commit()
                    response = f"Salve, sono {character_name}. Sono pronto per questa intervista impossibile. Cosa vorresti chiedermi?"
                else:
                    # For regular messages in interview mode, include context
                    message = f"[CONTESTO PERSONAGGIO]\n{system_prompt}\n\n[DOMANDA UTENTE]\n{message}"
        elif mode == 'rag':
            # Modalità RAG - Retrieval Augmented Generation
            files = context.get('files', [])
            if files:
                # Imposta il titolo della chat
                c.execute('UPDATE chatbot2_chats SET title = ? WHERE id = ?',
                         (f"Analisi documenti: {message[:30]}...", chat_id))
                conn.commit()
                
                # Estrai il contenuto dei documenti
                documents_content = []
                for file_id in files:
                    content = get_document_content(file_id)
                    if content:
                        # Ottieni il nome del file
                        c.execute('SELECT original_name, type FROM resources WHERE id = ?', (file_id,))
                        file_info = c.fetchone()
                        if file_info:
                            file_name, file_type = file_info
                            documents_content.append(f"--- Documento: {file_name} (Tipo: {file_type}) ---\n{content}\n")
                
                # Aggiungi il contenuto dei documenti al messaggio
                if documents_content:
                    documents_text = "\n\n".join(documents_content)
                    message = f"[DOCUMENTI]\n{documents_text}\n\n[DOMANDA]\n{message}"
        
        try:
            if model_name in ['gpt4o-mini', 'o3-mini', 'o3-mini-reasoning']:
                if not api_status['openai']:
                    raise Exception("OpenAI API not properly initialized. Please check your API key and try again.")
                
                # Use the selected model
                model_mapping = {
                    'gpt4o-mini': 'gpt-4-0125-preview',
                    'o3-mini': 'gpt-3.5-turbo',
                    'o3-mini-reasoning': 'gpt-3.5-turbo-instruct'
                }
                gpt_models = [model_mapping[model_name]]
                
                response = None
                last_error = None
                
                # Get chat history for context
                c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at ASC', (chat_id,))
                chat_history = c.fetchall()
                
                # Prepare messages array with chat history
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add chat history in reverse order (oldest first)
                for role, content in reversed(chat_history):
                    if role == 'user':
                        messages.append({"role": "user", "content": content})
                    elif role == 'bot':
                        messages.append({"role": "assistant", "content": content})
                
                # Add current user message
                messages.append({"role": "user", "content": message})
                
                for model in gpt_models:
                    try:
                        completion = openai_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=4096
                        )
                        
                        if completion and completion.choices:
                            response = completion.choices[0].message.content
                            print(f"Successfully used {model}")
                            break
                            
                    except Exception as e:
                        error_msg = str(e)
                        print(f"{model} error: {error_msg}")
                        last_error = e
                        
                        if 'rate_limit' in error_msg.lower():
                            raise Exception("Rate limit exceeded. Please try again in a few seconds.")
                        elif 'invalid_request_error' in error_msg.lower():
                            raise Exception("Invalid request to OpenAI API. Please check your input.")
                        elif 'authentication' in error_msg.lower():
                            raise Exception("Authentication failed. Please check your OpenAI API key.")
                        
                        # Continue to next model if this one failed
                        continue
                
                if not response:
                    if last_error:
                        raise Exception(f"OpenAI API Error: {str(last_error)}")
                    else:
                        raise Exception("All GPT models failed to generate a response.")
                
            elif model_name.startswith('ollama:'):
                # Extract the actual model name from the 'ollama:model_name' format
                ollama_model = model_name.split(':', 1)[1] if ':' in model_name else ''
                
                if not ollama_model:
                    raise Exception("No Ollama model specified. Format should be 'ollama:model_name'")
                
                print(f"Processing request for Ollama model: {ollama_model}")
                
                # Global rate limiting for Ollama API
                global ollama_last_request_time
                current_time = time.time()
                time_since_last_request = current_time - ollama_last_request_time
                
                if time_since_last_request < OLLAMA_RATE_LIMIT_DELAY:
                    # Need to wait before making another request
                    wait_time = OLLAMA_RATE_LIMIT_DELAY - time_since_last_request
                    print(f"Rate limiting: Waiting {wait_time:.2f} seconds before making Ollama API request")
                    time.sleep(wait_time)
                
                # Check if Ollama is running
                if not api_status['ollama']:
                    print("Ollama API status check failed, rechecking...")
                    # Try to check Ollama availability again
                    try:
                        response = requests.get(f"{OLLAMA_API_ENDPOINT}/api/tags", timeout=3)
                        if response.status_code == 200:
                            api_status['ollama'] = True
                            print("Ollama API is now available")
                            ollama_last_request_time = time.time()
                        else:
                            print(f"Ollama API check failed with status code: {response.status_code}")
                            raise Exception(f"Ollama API returned status code: {response.status_code}")
                    except Exception as e:
                        print(f"Ollama API check failed with error: {str(e)}")
                        raise Exception(f"Ollama non è in esecuzione o non è raggiungibile: {str(e)}")
                
                try:
                    # Get chat history for context
                    c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at ASC', (chat_id,))
                    chat_history = c.fetchall()
                    
                    # Prepare messages array with chat history
                    messages = []
                    
                    # Add system message if provided
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    
                    # Add chat history in reverse order (oldest first)
                    for role, content in reversed(chat_history):
                        if role == 'user':
                            messages.append({"role": "user", "content": content})
                        elif role == 'bot' or role == 'assistant':
                            messages.append({"role": "assistant", "content": content})
                    
                    # Add current user message
                    messages.append({"role": "user", "content": message})
                    
                    print(f"Prepared {len(messages)} messages for Ollama API request")
                    
                    # Maximum number of retries for rate limiting
                    max_retries = 5  # Increased from 3 to 5
                    retry_count = 0
                    retry_delay = 3  # seconds - increased from 2
                    
                    while retry_count < max_retries:
                        try:
                            print(f"Making Ollama API request (attempt {retry_count + 1}/{max_retries})")
                            
                            # Update last request time
                            ollama_last_request_time = time.time()
                            
                            # Make request to Ollama API
                            ollama_response = requests.post(
                                f"{OLLAMA_API_ENDPOINT}/api/chat",
                                json={
                                    "model": ollama_model,
                                    "messages": messages,
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.7,
                                        "num_predict": 1024  # Similar to max_tokens
                                    }
                                },
                                timeout=120  # Increased timeout for model inference
                            )
                            
                            print(f"Ollama API response status code: {ollama_response.status_code}")
                            
                            # Check for rate limit errors
                            if ollama_response.status_code == 429:
                                print(f"Ollama rate limit exceeded, retrying in {retry_delay} seconds...")
                                retry_count += 1
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            
                            # Check for other errors
                            if ollama_response.status_code != 200:
                                error_text = ollama_response.text
                                print(f"Ollama API error: {error_text}")
                                raise Exception(f"Ollama API returned status code: {ollama_response.status_code}, message: {error_text}")
                            
                            # Parse response
                            result = ollama_response.json()
                            print(f"Ollama API response format: {list(result.keys())}")
                            
                            if 'message' in result and 'content' in result['message']:
                                response = result['message']['content']
                                print(f"Successfully used Ollama model: {ollama_model}")
                                break
                            else:
                                print(f"Unexpected Ollama API response format: {result}")
                                raise Exception(f"Unexpected response format from Ollama API: {result}")
                            
                        except requests.exceptions.RequestException as e:
                            print(f"Ollama API request exception: {str(e)}")
                            if 'timeout' in str(e).lower():
                                if retry_count < max_retries - 1:
                                    print(f"Timeout error, retrying in {retry_delay} seconds...")
                                    retry_count += 1
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    raise Exception(f"Ollama API request timed out after {retry_count} attempts. The model might be too large or the server is busy.")
                            else:
                                raise Exception(f"Error communicating with Ollama API: {str(e)}")
                        
                        except Exception as e:
                            error_msg = str(e)
                            print(f"Ollama error: {error_msg}")
                            
                            if any(term in error_msg.lower() for term in ['rate limit', 'rate_limit', '429']):
                                if retry_count < max_retries - 1:
                                    print(f"Rate limit error, retrying in {retry_delay} seconds...")
                                    retry_count += 1
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    raise Exception("Rate limit exceeded after multiple retries. Please try again in a few seconds.")
                            else:
                                raise
                    
                    if not response:
                        raise Exception(f"Failed to get a response from Ollama model: {ollama_model}")
                    
                except Exception as e:
                    print(f"Ollama API Error: {str(e)}")
                    raise Exception(f"Error with OLLAMA API: {str(e)}")
                    
            elif model_name == 'claude':
                if not anthropic_api_key:
                    raise Exception("Anthropic API key not configured. Please check your environment variables.")

                try:
                    # Get chat history for context
                    c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at ASC', (chat_id,))
                    chat_history = c.fetchall()
                    
                    # Prepare messages array with chat history
                    messages = []
                    
                    # Add chat history in reverse order (oldest first)
                    for role, content in reversed(chat_history):
                        if role == 'user':
                            messages.append({"role": "user", "content": content})
                        elif role == 'bot':
                            messages.append({"role": "assistant", "content": content})
                    
                    # Add current user message
                    messages.append({"role": "user", "content": message})

                    # Define models to try in order
                    claude_models = [
                        "claude-3",
                        "claude-3-mini",
                        "claude-3-reasoning"
                    ]

                    response = None
                    last_error = None

                    for model in claude_models:
                        try:
                            # Create messages request without system role
                            try:
                                # Try newer API format first
                                message_response = anthropic_client.messages.create(
                                    model=model,
                                    max_tokens=4096,
                                    messages=messages,
                                    temperature=0.7,
                                    system=system_prompt if system_prompt else None
                                )
                            except (TypeError, AttributeError) as api_error:
                                print(f"Falling back to alternative Anthropic API format: {str(api_error)}")
                                # Try alternative API format (for older versions)
                                import pkg_resources
                                anthropic_version = pkg_resources.get_distribution("anthropic").version
                                
                                if hasattr(anthropic_client, 'completions'):
                                    # Old API format (pre-0.5.0)
                                    prompt = f"\n\nHuman: {message}\n\nAssistant:"
                                    message_response = anthropic_client.completions.create(
                                        model=model,
                                        prompt=prompt,
                                        max_tokens_to_sample=4096,
                                        temperature=0.7
                                    )
                                    # Convert to expected format
                                    class Content:
                                        def __init__(self, text):
                                            self.text = text
                                    
                                    class Response:
                                        def __init__(self, content):
                                            self.content = [Content(content)]
                                    
                                    message_response = Response(message_response.completion)

                            if message_response and message_response.content:
                                response = message_response.content[0].text
                                print(f"Successfully used {model}")
                                break

                        except Exception as e:
                            error_msg = str(e)
                            print(f"{model} error: {error_msg}")
                            last_error = e

                            if 'rate_limit_error' in error_msg:
                                raise Exception("Rate limit exceeded. Please try again in a few seconds.")
                            elif 'invalid_request_error' in error_msg:
                                raise Exception("Invalid request to Claude API. Please check your input.")
                            elif 'authentication_error' in error_msg:
                                raise Exception("Authentication failed. Please check your Claude API key.")
                            
                            # Continue to next model if this one failed
                            continue

                    if not response:
                        if last_error:
                            raise Exception(f"Claude API Error: {str(last_error)}")
                        else:
                            raise Exception("All Claude models failed to generate a response.")
                        
                except Exception as e:
                    print(f"Claude API Error: {str(e)}")
                    raise Exception(f"Claude API Error: {str(e)}")
                
            else:  # gemini
                if not api_status['gemini']:
                    raise Exception("Gemini API not properly initialized. Please check your API key and try again.")
                
                # Define Gemini models to try in order
                gemini_models = [
                    "gemini-1.5-pro-001",
                    "gemini-1.0-pro-001",
                    "gemini-pro"
                ]
                
                response = None
                last_error = None
                
                # Get chat history for context
                c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at ASC', (chat_id,))
                chat_history = c.fetchall()
                
                # Build conversation history
                conversation = ""
                if system_prompt:
                    conversation += f"System: {system_prompt}\n\n"
                
                # Add chat history in reverse order (oldest first)
                for role, content in reversed(chat_history):
                    if role == 'user':
                        conversation += f"User: {content}\n"
                    elif role == 'bot':
                        conversation += f"Assistant: {content}\n"
                
                # Add current message
                conversation += f"User: {message}\n"
                
                for model_version in gemini_models:
                    try:
                        model = genai.GenerativeModel(model_version)
                        
                        # Generate response with full conversation context
                        result = model.generate_content(conversation)
                        
                        if result and result.text:
                            response = result.text
                            print(f"Successfully used {model_version}")
                            break
                            
                    except Exception as e:
                        error_msg = str(e)
                        print(f"{model_version} error: {error_msg}")
                        last_error = e
                        
                        if 'rate' in error_msg.lower() and 'limit' in error_msg.lower():
                            raise Exception("Rate limit exceeded. Please try again in a few seconds.")
                        elif 'invalid' in error_msg.lower():
                            raise Exception("Invalid request to Gemini API. Please check your input.")
                        elif any(term in error_msg.lower() for term in ['api key', 'authentication', 'permission']):
                            raise Exception("Authentication failed. Please check your Gemini API key.")
                        elif 'quota' in error_msg.lower():
                            raise Exception("Gemini API quota exceeded. Please try again later.")
                        
                        # Continue to next model if this one failed
                        continue
                
                if not response:
                    if last_error:
                        raise Exception(f"Gemini API Error: {str(last_error)}")
                    else:
                        raise Exception("All Gemini models failed to generate a response.")
                
        except Exception as e:
            error_msg = str(e)
            print(f"AI Model error ({model_name}): {error_msg}")
            if model_name == 'claude' and 'not_found_error' in error_msg:
                raise Exception(f"Claude API Error: Invalid model name. Please check your API configuration.")
            elif 'api_key' in error_msg.lower():
                raise Exception(f"API Key Error: Please check your {model_name.upper()} API key configuration.")
            else:
                raise Exception(f"Error with {model_name.upper()} API: {error_msg}")
        
        if not response:
            raise Exception("No response generated from the AI model")
    
        if response is None:
            raise Exception("No response generated from the AI model")
            
        # Per la modalità interrogazione, gestiamo in modo speciale sia il messaggio START_INTERROGATION
        # che tutte le altre risposte per assicurarci che terminino con una domanda
        if context.get('mode') == 'interrogazione':
            if message == 'START_INTERROGATION':
                # Questo è il messaggio iniziale per avviare la modalità interrogazione
                # La risposta speciale è già gestita sopra dove impostiamo la risposta iniziale
                print("DEBUG - Starting interrogation mode with initial prompt")
                # Nessuna elaborazione aggiuntiva necessaria - il system prompt contiene già le istruzioni
                pass
            else:
                # Per tutti gli altri messaggi, verifichiamo che la risposta termini con una domanda
                # Se non termina con una domanda, ne aggiungiamo una
                
                # Verifichiamo se la risposta termina già con un punto interrogativo
                ends_with_question = False
                
                # Controlliamo se la risposta termina con un punto interrogativo
                # o contiene frasi interrogative alla fine
                if '?' in response[-20:]:  # Controlliamo gli ultimi 20 caratteri
                    ends_with_question = True
                
                # Se non termina con una domanda, aggiungiamo una domanda appropriata
                if not ends_with_question:
                    print("DEBUG - Adding question to response in interrogation mode")
                    
                    # Utilizziamo il modello per generare una domanda pertinente
                    if model_name.startswith('gpt'):
                        # Per i modelli OpenAI
                        completion = openai_client.chat.completions.create(
                            model="gpt-4" if model_name == 'gpt4' else "gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": f"Sei un insegnante di {context.get('subject')} che sta conducendo un'interrogazione. L'argomento è: {context.get('systemPrompt', '').split('L\'argomento è:')[1].split('.')[0] if 'L\'argomento è:' in context.get('systemPrompt', '') else 'argomento generale'}. Devi formulare una domanda pertinente per continuare l'interrogazione."}, 
                                {"role": "user", "content": f"Ho appena risposto all'utente con: '{response}'. Genera una domanda pertinente per continuare l'interrogazione. La domanda deve essere breve (massimo 1-2 righe) e deve essere formulata in modo da continuare naturalmente la conversazione. Rispondi SOLO con la domanda, senza introduzioni o spiegazioni."}
                            ]
                        )
                        question = completion.choices[0].message.content.strip()
                        
                        # Assicuriamoci che la domanda termini con un punto interrogativo
                        if not question.endswith('?'):
                            question += '?'
                            
                        # Aggiungiamo la domanda alla risposta
                        response = response.rstrip() + "\n\n" + question
                    elif model_name.startswith('claude'):
                        # Per i modelli Claude
                        try:
                            message_request = {
                                "model": "claude-3-haiku-20240307",
                                "max_tokens": 100,
                                "messages": [
                                    {"role": "system", "content": f"Sei un insegnante di {context.get('subject')} che sta conducendo un'interrogazione. Devi formulare una domanda pertinente per continuare l'interrogazione."}, 
                                    {"role": "user", "content": f"Ho appena risposto all'utente con: '{response}'. Genera una domanda pertinente per continuare l'interrogazione. La domanda deve essere breve (massimo 1-2 righe) e deve essere formulata in modo da continuare naturalmente la conversazione. Rispondi SOLO con la domanda, senza introduzioni o spiegazioni."}
                                ]
                            }
                            
                            anthropic_response = anthropic_client.messages.create(**message_request)
                            question = anthropic_response.content[0].text.strip()
                            
                            # Assicuriamoci che la domanda termini con un punto interrogativo
                            if not question.endswith('?'):
                                question += '?'
                                
                            # Aggiungiamo la domanda alla risposta
                            response = response.rstrip() + "\n\n" + question
                        except Exception as e:
                            print(f"Error generating question with Claude: {str(e)}")
                            # Fallback: aggiungiamo una domanda generica
                            response = response.rstrip() + "\n\nCosa ne pensi di questo argomento? Hai altre domande?"
                    else:
                        # Per altri modelli, aggiungiamo una domanda generica
                        response = response.rstrip() + "\n\nCosa ne pensi di questo argomento? Hai altre domande?"
            
        # Conteggio token per l'input e l'output
        # Ottieni i messaggi rilevanti per il conteggio
        c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at ASC', (chat_id,))
        chat_messages = [{'role': 'user' if msg[0] == 'user' else 'assistant', 'content': msg[1]} for msg in c.fetchall()]
        
        # Aggiungi l'ultimo messaggio dell'utente se non è già incluso
        if not any(msg.get('role') == 'user' and msg.get('content') == message for msg in chat_messages):
            chat_messages.append({'role': 'user', 'content': message})
        
        # Calcola token per input
        input_messages = [msg for msg in chat_messages if msg['role'] == 'user']
        input_tokens = count_tokens(input_messages)
        
        # Calcola token per output
        output_messages = [msg for msg in chat_messages if msg['role'] == 'assistant']
        output_messages.append({'role': 'assistant', 'content': response})  # Aggiungi la risposta attuale
        output_tokens = count_tokens(output_messages)
        
        # Calcola il totale
        total_tokens = input_tokens + output_tokens
        
        print(f"DEBUG - Input tokens: {input_tokens}")
        print(f"DEBUG - Output tokens: {output_tokens}")
        print(f"DEBUG - Total tokens: {total_tokens}")
        
        # Informa l'utente del conteggio dei token
        token_info = {
            'total': total_tokens,
            'input': input_tokens,
            'output': output_tokens
        }
        
        # Save AI response
        c.execute('''INSERT INTO chatbot2_messages (chat_id, role, content, model)
                     VALUES (?, ?, ?, ?)''',
                 (chat_id, 'assistant', response, context.get('model')))
        
        conn.commit()
        
        return jsonify({
            'response': response,
            'chatId': chat_id,
            'token_info': token_info
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        if conn:
            conn.rollback()
            return jsonify({'error': str(e)}), 500
            
    finally:
        if conn:
            conn.close()
