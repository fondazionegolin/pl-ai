import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime

# Inizializza Firebase Admin SDK
import os

# Ottieni il percorso assoluto del file delle credenziali
cred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'firebase-credentials.json')
cred = credentials.Certificate(cred_path)

# Inizializza Firebase solo se non è già stato inizializzato
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()

class FirebaseDB:
    @staticmethod
    def init_collections():
        """Verifica che le collezioni esistano."""
        # Non necessario in Firestore, le collezioni vengono create automaticamente

    @staticmethod
    def save_chat(chat_id: str, messages: list, user_id: str, title: str = None):
        """Salva una chat su Firestore."""
        chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_id)
        chat_ref.set({
            'id': chat_id,
            'messages': json.dumps(messages),
            'title': title,
            'user_id': user_id,
            'created_at': firestore.SERVER_TIMESTAMP
        })

    @staticmethod
    def get_chat(chat_id: str, user_id: str):
        """Recupera una chat specifica."""
        chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_id)
        chat = chat_ref.get()
        if chat.exists:
            data = chat.to_dict()
            data['messages'] = json.loads(data['messages'])
            return data
        return None

    @staticmethod
    def get_all_chats(user_id: str):
        """Recupera tutte le chat di un utente specifico."""
        chats = db.collection('users').document(user_id).collection('chats')\
                .order_by('created_at', direction=firestore.Query.DESCENDING).stream()
        return [{**chat.to_dict(), 'messages': json.loads(chat.to_dict()['messages'])} for chat in chats]

    @staticmethod
    def save_learning_unit(unit_id: str, title: str, content: str, quiz: str = None, answers: list = None):
        """Salva un'unità di apprendimento."""
        unit_ref = db.collection('learning_units').document(unit_id)
        unit_ref.set({
            'id': unit_id,
            'title': title,
            'content': content,
            'quiz': quiz,
            'answers': json.dumps(answers or []),
            'total_questions': 0,
            'correct_answers': 0
        })

    @staticmethod
    def get_learning_unit(unit_id: str):
        """Recupera un'unità di apprendimento specifica."""
        unit_ref = db.collection('learning_units').document(unit_id)
        unit = unit_ref.get()
        if unit.exists:
            data = unit.to_dict()
            data['answers'] = json.loads(data['answers'])
            return data
        return None

    @staticmethod
    def get_all_learning_units():
        """Recupera tutte le unità di apprendimento."""
        units = db.collection('learning_units').stream()
        return [{**unit.to_dict(), 'answers': json.loads(unit.to_dict()['answers'])} for unit in units]

    @staticmethod
    def update_learning_unit_stats(unit_id: str, total_questions: int, correct_answers: int):
        """Aggiorna le statistiche di un'unità di apprendimento."""
        unit_ref = db.collection('learning_units').document(unit_id)
        unit_ref.update({
            'total_questions': total_questions,
            'correct_answers': correct_answers
        })
