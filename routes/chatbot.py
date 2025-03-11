from flask import Blueprint, render_template, request, jsonify, session
from openai import OpenAI
import os
import json
import sqlite3
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from routes.db import update_api_credits

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

chatbot = Blueprint('chatbot', __name__)

# Inizializza il database
def init_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id TEXT PRIMARY KEY,
                  messages TEXT,
                  title TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Chiama init_db all'avvio
init_db()

def get_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.sqlite')
    return sqlite3.connect(db_path)

@chatbot.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@chatbot.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        chat_id = data.get('chat_id')
        
        # Calcola il numero di token nell'input
        input_tokens = count_tokens(messages)
        print(f"DEBUG - Input tokens: {input_tokens}")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Calcola il numero di token nell'output
        output_content = response.choices[0].message.content
        output_tokens = count_tokens([{"role": "assistant", "content": output_content}])
        print(f"DEBUG - Output tokens: {output_tokens}")
        
        # Calcola il totale dei token utilizzati
        total_tokens = input_tokens + output_tokens
        print(f"DEBUG - Total tokens: {total_tokens}")
        
        # Aggiorna i crediti API se l'utente è loggato, passando il numero di token
        if session.get('user_id') and session.get('username'):
            username = session.get('username')
            print(f"DEBUG - Updating API credits for user: {username} with {total_tokens} tokens")
            result = update_api_credits(username, 'openai', amount=total_tokens)
            print(f"DEBUG - Update result: {result}")
        
        # Aggiungi la risposta dell'assistente ai messaggi con informazioni sui token
        token_info = f"\n\n---\nToken utilizzati: {total_tokens} (Input: {input_tokens}, Output: {output_tokens})"
        messages.append({"role": "assistant", "content": response.choices[0].message.content + token_info})
        
        # Se abbiamo un chat_id, salviamo la conversazione
        if chat_id:
            conn = get_db()
            c = conn.cursor()
            c.execute('UPDATE chats SET messages = ?, title = ? WHERE id = ?',
                     (json.dumps(messages), get_chat_title(messages), chat_id))
            conn.commit()
            conn.close()
        
        return jsonify({
            'message': response.choices[0].message.content + token_info,
            'timestamp': datetime.now().isoformat(),
            'token_info': {
                'total': total_tokens,
                'input': input_tokens,
                'output': output_tokens
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/initialize_chat', methods=['POST'])
def initialize_chat():
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        
        # Crea un nuovo chat_id
        chat_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Ciao! Sono pronto ad aiutarti con il machine learning!"}
        ]
        
        # Salva la nuova chat nel database
        conn = get_db()
        c = conn.cursor()
        c.execute('INSERT INTO chats (id, messages, title) VALUES (?, ?, ?)',
                 (chat_id, json.dumps(messages), "Nuova conversazione"))
        conn.commit()
        conn.close()
        
        return jsonify({
            'messages': messages,
            'initial_message': messages[-1]['content'],
            'chat_id': chat_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/load_chats', methods=['GET'])
def load_chats():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT id, messages, title, created_at FROM chats ORDER BY created_at DESC')
        chats = []
        for row in c.fetchall():
            chats.append({
                'id': row[0],
                'title': row[2] or get_chat_title(json.loads(row[1])),
                'timestamp': row[3]
            })
        conn.close()
        return jsonify(chats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/load_chat/<chat_id>', methods=['GET'])
def load_chat(chat_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT messages FROM chats WHERE id = ?', (chat_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return jsonify({'messages': json.loads(result[0])})
        return jsonify({'error': 'Chat non trovata'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def count_tokens(messages):
    """Conta il numero di token in una lista di messaggi."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = 0
        for message in messages:
            # Ogni messaggio ha un overhead di 4 token
            num_tokens += 4
            # Aggiungi token per ogni campo nel messaggio
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                # Ogni nome di campo (role, content) costa 1 token
                num_tokens += 1
        # Ogni richiesta ha un overhead di 2 token
        num_tokens += 2
        return num_tokens
    except Exception as e:
        print(f"Errore nel conteggio dei token: {str(e)}")
        # In caso di errore, restituisci una stima approssimativa basata sui caratteri
        return sum(len(msg.get('content', '')) for msg in messages) // 4

def get_chat_title(messages):
    # Cerca il primo messaggio dell'utente per usarlo come titolo
    for msg in messages:
        if msg['role'] == 'user':
            # Prendi le prime 30 caratteri del messaggio
            return msg['content'][:30] + '...' if len(msg['content']) > 30 else msg['content']
    return 'Nuova conversazione'

@chatbot.route('/update_system_prompt', methods=['POST'])
def update_system_prompt():
    try:
        data = request.json
        chat_id = data.get('chat_id')
        system_prompt = data.get('system_prompt', '')
        
        if not chat_id:
            return jsonify({'error': 'ID chat mancante'}), 400
            
        # Carica i messaggi esistenti
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT messages FROM chats WHERE id = ?', (chat_id,))
        result = c.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'error': 'Chat non trovata'}), 404
            
        messages = json.loads(result[0])
        
        # Aggiorna o aggiungi il messaggio di sistema
        system_message_found = False
        for i, msg in enumerate(messages):
            if msg['role'] == 'system':
                messages[i]['content'] = system_prompt
                system_message_found = True
                break
                
        if not system_message_found:
            # Aggiungi un nuovo messaggio di sistema all'inizio
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Salva i messaggi aggiornati
        c.execute('UPDATE chats SET messages = ? WHERE id = ?',
                 (json.dumps(messages), chat_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'messages': messages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
