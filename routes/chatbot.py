from flask import Blueprint, render_template, request, jsonify
from openai import OpenAI
import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

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
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Aggiungi la risposta dell'assistente ai messaggi
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        # Se abbiamo un chat_id, salviamo la conversazione
        if chat_id:
            conn = get_db()
            c = conn.cursor()
            c.execute('UPDATE chats SET messages = ?, title = ? WHERE id = ?',
                     (json.dumps(messages), get_chat_title(messages), chat_id))
            conn.commit()
            conn.close()
        
        return jsonify({
            'message': response.choices[0].message.content,
            'timestamp': datetime.now().isoformat()
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

def get_chat_title(messages):
    # Cerca il primo messaggio dell'utente per usarlo come titolo
    for msg in messages:
        if msg['role'] == 'user':
            # Prendi le prime 30 caratteri del messaggio
            return msg['content'][:30] + '...' if len(msg['content']) > 30 else msg['content']
    return 'Nuova conversazione'
