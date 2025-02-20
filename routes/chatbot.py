from flask import Blueprint, render_template, request, jsonify
from routes.auth import login_required, verify_token_and_get_user
from openai import OpenAI
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from routes.firebase_db import FirebaseDB

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

chatbot = Blueprint('chatbot', __name__)

# Inizializza il database
FirebaseDB.init_collections()

@chatbot.route('/chatbot')
@login_required
def chatbot_page():
    pass
    return render_template('chatbot.html')

@chatbot.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        chat_id = data.get('chat_id')
        pass
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Aggiungi la risposta dell'assistente ai messaggi
        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        # Se abbiamo un chat_id, salviamo la conversazione
        if chat_id:
            # Ottieni l'ID dell'utente dal token
            user_data = verify_token_and_get_user()
            if not user_data:
                return jsonify({'error': 'Utente non autenticato'}), 401
            
            FirebaseDB.save_chat(chat_id, messages, user_data['uid'], get_chat_title(messages))
        
        return jsonify({
            'message': response.choices[0].message.content,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/initialize_chat', methods=['POST'])
@login_required
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
        
        # Ottieni l'ID dell'utente dal token
        user_data = verify_token_and_get_user()
        if not user_data:
            return jsonify({'error': 'Utente non autenticato'}), 401
            
        # Salva la nuova chat su Firebase
        FirebaseDB.save_chat(chat_id, messages, user_data['uid'], "Nuova conversazione")
        
        return jsonify({
            'messages': messages,
            'initial_message': messages[-1]['content'],
            'chat_id': chat_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/load_chats', methods=['GET'])
@login_required
def load_chats():
    try:
        # Ottieni l'ID dell'utente dal token
        user_data = verify_token_and_get_user()
        if not user_data:
            return jsonify({'error': 'Utente non autenticato'}), 401
            
        chats = FirebaseDB.get_all_chats(user_data['uid'])
        formatted_chats = [{
            'id': chat['id'],
            'title': chat['title'] or get_chat_title(chat['messages']),
            'timestamp': chat['created_at']
        } for chat in chats]
        return jsonify(formatted_chats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot.route('/load_chat/<chat_id>', methods=['GET'])
@login_required
def load_chat(chat_id):
    try:
        # Ottieni l'ID dell'utente dal token
        user_data = verify_token_and_get_user()
        if not user_data:
            return jsonify({'error': 'Utente non autenticato'}), 401
            
        chat = FirebaseDB.get_chat(chat_id, user_data['uid'])
        if chat:
            return jsonify({'messages': chat['messages']})
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
