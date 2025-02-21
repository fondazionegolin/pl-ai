from flask import Blueprint, request, jsonify, render_template, current_app
from routes.auth import login_required, get_current_user, update_user_profile, allowed_file
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
from datetime import datetime

profile = Blueprint('profile', __name__)

@profile.route('/settings')
@login_required
def settings():
    try:
        # Ottieni i dati dell'utente corrente
        user_data = get_current_user()
        if not user_data:
            return jsonify({'error': 'Utente non trovato'}), 404
            
        return render_template('settings.html',
                             user=user_data,
                             firebase_api_key=os.getenv('FIREBASE_API_KEY'),
                             firebase_auth_domain=os.getenv('FIREBASE_AUTH_DOMAIN'),
                             firebase_project_id=os.getenv('FIREBASE_PROJECT_ID'),
                             firebase_app_id=os.getenv('FIREBASE_APP_ID'))
                             
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@profile.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Ottiene il profilo dell'utente corrente."""
    try:
        user_data = get_current_user()
        if not user_data:
            return jsonify({'error': 'Utente non trovato'}), 404
        return jsonify(user_data)
    except Exception as e:
        print(f"Errore nel recupero del profilo: {str(e)}")
        return jsonify({'error': 'Errore interno del server'}), 500

@profile.route('/api/profile', methods=['POST'])
@login_required
def update_profile():
    try:
        print('[DEBUG] Request data:', request.get_json())
        # Ottieni i dati dell'utente corrente
        user_data = get_current_user()
        if not user_data:
            return jsonify({'error': 'Utente non trovato'}), 404
            
        # Ottieni i dati dal form
        data = request.json
        bio = data.get('bio', '').strip()
        
        # Validazione dei dati
        if len(bio) > 500:
            return jsonify({'error': 'La bio non può superare i 500 caratteri'}), 400
            
        # Aggiorna il profilo
        profile_data = {
            'bio': bio,
            'updated_at': datetime.now().isoformat()
        }
        
        update_user_profile(user_data['uid'], profile_data)
        
        return jsonify({
            'message': 'Profilo aggiornato con successo',
            'profile': profile_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@profile.route('/api/profile/picture', methods=['POST'])
@login_required
def update_profile_picture():
    try:
        # Ottieni i dati dell'utente corrente
        user_data = get_current_user()
        if not user_data:
            return jsonify({'error': 'Utente non trovato'}), 404
            
        if 'profile_picture' not in request.files:
            return jsonify({'error': 'Nessun file caricato'}), 400
            
        file = request.files['profile_picture']
        
        if file.filename == '':
            return jsonify({'error': 'Nessun file selezionato'}), 400
            
        if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'gif'}):
            # Processa l'immagine
            img = Image.open(file)
            
            # Ridimensiona l'immagine mantenendo l'aspect ratio
            max_size = (300, 300)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Converti in RGB se necessario
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Salva l'immagine
            filename = secure_filename(f"{user_data['uid']}_profile.jpg")
            upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], 'profile_pictures')
            os.makedirs(upload_folder, exist_ok=True)
            
            img_path = os.path.join(upload_folder, filename)
            img.save(img_path, 'JPEG', quality=85)
            
            # Aggiorna il profilo con il nuovo percorso dell'immagine
            profile_data = {
                'profile_picture': f"/uploads/profile_pictures/{filename}"
            }
            
            update_user_profile(user_data['uid'], profile_data)
            
            return jsonify({
                'message': 'Immagine del profilo aggiornata con successo',
                'profile_picture_url': profile_data['profile_picture']
            })
            
        return jsonify({'error': 'Tipo di file non supportato'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
