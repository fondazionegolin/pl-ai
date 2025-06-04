from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
import json
import re
import os
from markdown import markdown
from bs4 import BeautifulSoup
import uuid
from datetime import datetime
from .db import get_user_db, login_required, get_user_db_path, init_user_db
from .openai_client import client

learning = Blueprint('learning', __name__)

@learning.route('/learning')
@login_required
def learning_page():
    return render_template('learning.html')

@learning.route('/generate_lesson', methods=['POST'])
@login_required
def generate_lesson():
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip() if data else ''
        if not topic:
            return jsonify({'success': False, 'error': 'L\'argomento è obbligatorio.'}), 400
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'error': 'Utente non autenticato.'}), 401
        # Prompt OpenAI per la mini-lezione
        prompt_lesson = f"Scrivi una mini-lezione (max 15 righe) sull'argomento: {topic}. Usa un linguaggio chiaro e didattico."
        lesson_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_lesson}]
        )
        lesson_content = lesson_resp.choices[0].message.content.strip()
        # Prompt OpenAI per le domande
        prompt_quiz = f"Crea 5 domande a risposta multipla (con 4 opzioni ciascuna) basate SOLO sulla seguente mini-lezione. Le domande devono essere specifiche e non generiche. Indica la risposta corretta per ciascuna domanda. Restituisci un JSON con una lista di oggetti: question, options (array), correct_index (int). Lezione:\n{lesson_content}"
        quiz_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_quiz}]
        )
        import json as pyjson
        import re
        quiz_json = re.search(r'\{.*\}|\[.*\]', quiz_resp.choices[0].message.content, re.DOTALL)
        if quiz_json:
            quiz = pyjson.loads(quiz_json.group())
        else:
            return jsonify({'success': False, 'error': 'Errore parsing quiz.'}), 500
        # Salva nel DB
        conn = get_user_db(username)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO topics (title, content, created_at, status, quiz) VALUES (?, ?, ?, ?, ?)',
            (topic, lesson_content, datetime.now().isoformat(), 'completed', pyjson.dumps(quiz)))
        topic_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'lesson': {'id': topic_id, 'title': topic, 'content': lesson_content}, 'quiz': quiz})
    except Exception as e:
        import traceback
        print('Errore in generate_lesson:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@learning.route('/lessons')
@login_required
def lessons():
    username = session.get('username')
    conn = get_user_db(username)
    cursor = conn.cursor()
    # Recupera id, title, quiz, answers dalla tabella topics (o learning_units se serve)
    cursor.execute('SELECT id, title, quiz FROM topics ORDER BY id DESC')
    lessons = []
    import json as pyjson
    for row in cursor.fetchall():
        lesson_id, title, quiz_json = row
        status = 'in_progress'
        try:
            quiz = pyjson.loads(quiz_json) if quiz_json else []
            # Se il quiz esiste ed è stato completato (tutte le domande corrette), status done
            # In questa versione semplice, se la lezione ha un quiz, la consideriamo 'done' se l'utente ha risposto a tutte e tutte giuste
            # Per ora, se il quiz esiste, mettiamo 'done', altrimenti 'in_progress'
            if quiz and isinstance(quiz, list) and len(quiz) > 0 and all('correct_index' in q for q in quiz):
                status = 'done'
        except Exception:
            status = 'in_progress'
        lessons.append({'id': lesson_id, 'title': title, 'status': status})
    conn.close()
    return jsonify({'lessons': lessons})

@learning.route('/lesson/<int:lesson_id>')
@login_required
def lesson_detail(lesson_id):
    username = session.get('username')
    conn = get_user_db(username)
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, content, quiz FROM topics WHERE id=?', (lesson_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Lezione non trovata'}), 404
    import json as pyjson
    quiz = pyjson.loads(row[3]) if row[3] else []
    return jsonify({'lesson': {'id': row[0], 'title': row[1], 'content': row[2]}, 'quiz': quiz})

@learning.route('/generate_approfondimenti', methods=['POST'])
@login_required
def generate_approfondimenti():
    try:
        data = request.get_json()
        title = data.get('title', '')
        content = data.get('content', '')
        
        if not title or not content:
            return jsonify({'success': False, 'error': 'Titolo o contenuto mancante'}), 400
        
        # Prompt per OpenAI per generare approfondimenti
        prompt = f"""Basandoti sulla seguente lezione dal titolo '{title}', genera 3-4 possibili approfondimenti correlati.
        Ogni approfondimento deve avere un titolo breve (massimo 5-6 parole) e una breve descrizione (2-3 frasi).
        Gli approfondimenti devono essere correlati al tema principale ma esplorare aspetti diversi o complementari.
        
        Lezione:
        {content}
        
        Restituisci il risultato in formato JSON con la seguente struttura:
        [{{
            "title": "Titolo approfondimento",
            "content": "Contenuto dell'approfondimento"
        }}]
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Estrai il JSON dalla risposta
        import json as pyjson
        import re
        
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response.choices[0].message.content, re.DOTALL)
        if json_match:
            approfondimenti = pyjson.loads(json_match.group())
            return jsonify({'success': True, 'approfondimenti': approfondimenti})
        else:
            return jsonify({'success': False, 'error': 'Errore nel parsing degli approfondimenti'}), 500
            
    except Exception as e:
        import traceback
        print('Errore in generate_approfondimenti:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@learning.route('/generate_detailed_approfondimento', methods=['POST'])
@login_required
def generate_detailed_approfondimento():
    try:
        data = request.get_json()
        title = data.get('title', '')
        lesson_title = data.get('lesson_title', '')
        lesson_content = data.get('lesson_content', '')
        
        if not title or not lesson_content:
            return jsonify({'success': False, 'error': 'Titolo o contenuto mancante'}), 400
        
        # Prompt per OpenAI per generare un approfondimento dettagliato
        prompt = f"""Basandoti sulla seguente lezione dal titolo '{lesson_title}', genera un approfondimento dettagliato sul tema specifico '{title}'.
        L'approfondimento deve essere completo, informativo e ben strutturato, con una lunghezza di almeno 300-400 parole.
        Includi informazioni rilevanti, esempi, e dettagli che espandono la comprensione dell'argomento.
        
        Lezione originale:
        {lesson_content}
        
        Titolo dell'approfondimento: {title}
        
        Fornisci un approfondimento completo e ben strutturato in formato HTML con paragrafi (<p>), titoli (<h3>, <h4>) e liste (<ul>, <li>) dove appropriato.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Ottieni il contenuto dell'approfondimento
        detailed_content = response.choices[0].message.content
        
        return jsonify({
            'success': True, 
            'title': title,
            'content': detailed_content
        })
            
    except Exception as e:
        import traceback
        print('Errore in generate_detailed_approfondimento:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@learning.route('/generate_quiz', methods=['POST'])
@login_required
def generate_quiz():
    try:
        data = request.get_json()
        lesson_id = data.get('lesson_id')
        approfondimenti = data.get('approfondimenti', [])
        
        if not lesson_id:
            return jsonify({'success': False, 'error': 'ID lezione mancante'}), 400
            
        # Recupera la lezione dal database
        username = session.get('username')
        conn = get_user_db(username)
        cursor = conn.cursor()
        cursor.execute('SELECT title, content FROM topics WHERE id=?', (lesson_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return jsonify({'success': False, 'error': 'Lezione non trovata'}), 404
            
        title, content = row
        
        # Prepara il contenuto completo includendo eventuali approfondimenti
        full_content = content
        if approfondimenti:
            full_content += "\n\nApprofondimenti:\n"
            for app in approfondimenti:
                if isinstance(app, dict) and 'title' in app:
                    # Usa text_content se disponibile (versione senza HTML), altrimenti usa content
                    app_content = app.get('text_content', app.get('content', ''))
                    # Rimuovi eventuali tag HTML residui
                    import re
                    app_content = re.sub(r'<[^>]*>', ' ', app_content)
                    full_content += f"\n{app['title']}:\n{app_content}\n"
        
        # Prompt OpenAI per le domande
        prompt_quiz = f"""Crea 5 domande a risposta multipla (con 4 opzioni ciascuna) basate sulla seguente mini-lezione e sui suoi approfondimenti. 
        Le domande devono essere specifiche e non generiche. 
        Indica la risposta corretta per ciascuna domanda. 
        Restituisci un JSON con una lista di oggetti con questa struttura: 
        [{{
            "question": "Testo della domanda",
            "options": ["Opzione 1", "Opzione 2", "Opzione 3", "Opzione 4"],
            "correct_index": 0 // indice della risposta corretta (0-3)
        }}]
        
        Lezione e approfondimenti:\n{full_content}"""
        
        quiz_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_quiz}]
        )
        
        import json as pyjson
        import re
        quiz_json = re.search(r'\{.*\}|\[.*\]', quiz_resp.choices[0].message.content, re.DOTALL)
        if quiz_json:
            quiz = pyjson.loads(quiz_json.group())
            
            # Salva il quiz nel database
            cursor.execute('UPDATE topics SET quiz = ? WHERE id = ?',
                (pyjson.dumps(quiz), lesson_id))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'quiz': quiz})
        else:
            conn.close()
            return jsonify({'success': False, 'error': 'Errore parsing quiz.'}), 500
            
    except Exception as e:
        import traceback
        print('Errore in generate_quiz:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


def clean_and_format_markdown(text):
    """Pulisce e formatta il testo in markdown."""
    if not text:
        return ""
    
    # Rimuovi spazi bianchi extra all'inizio e alla fine
    text = text.strip()
    
    # Assicurati che i titoli abbiano spazio dopo il cancelletto
    text = re.sub(r'(#+)([^\s#])', r'\1 \2', text)
    
    # Assicurati che i capitoli numerati siano formattati correttamente
    text = re.sub(r'##\s+(\d+)\.\s+', r'## \1. ', text)
    
    # Aggiungi classi CSS per migliorare la formattazione
    # Titolo principale
    text = re.sub(r'^#\s+(.*?)$', r'# \1 {.main-title}', text, flags=re.MULTILINE)
    
    # Sottotitoli
    text = re.sub(r'^##\s+(.*?)$', r'## \1 {.chapter-title}', text, flags=re.MULTILINE)
    
    # Evidenzia termini importanti
    text = re.sub(r'`([^`]+)`', r'<span class="highlight">\1</span>', text)
    
    # Aggiungi classi per paragrafi
    text = re.sub(r'(\n\n)([^#\n][^\n]+)', r'\1<p class="content-paragraph">\2</p>', text)
    
    return text

@learning.route('/topics', methods=['GET'])
@login_required
def get_topics():
    try:
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Recupera tutte le lezioni dell'utente
        c.execute('''
            SELECT id, title, total_questions, correct_answers 
            FROM learning_units 
            ORDER BY id DESC
        ''')
        
        results = c.fetchall()
        topics = []
        
        for row in results:
            topic_id, title, total_questions, correct_answers = row
            
            # Determina lo stato della lezione
            status = 'not_started'
            if total_questions > 0:
                if correct_answers == total_questions:
                    status = 'completed'
                elif correct_answers > 0:
                    status = 'in_progress'
            
            topics.append({
                'id': topic_id,
                'title': title,
                'status': status,
                'progress': {
                    'total': total_questions,
                    'correct': correct_answers
                }
            })
        
        conn.close()
        return jsonify({'topics': topics})
        
    except Exception as e:
        print(f"Errore nel recupero dei topics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@learning.route('/get_topic/<topic_id>', methods=['GET'])
@login_required
def get_topic(topic_id):
    try:
        # Converti topic_id in intero
        try:
            topic_id = int(topic_id)
        except ValueError:
            print(f"Errore: topic_id non valido: {topic_id}")
            return jsonify({'error': f'ID lezione non valido: {topic_id}. Deve essere un numero.'}), 400
        
        # Verifica che il database esista
        db_path = get_user_db_path(session['username'])
        if not os.path.exists(db_path):
            print(f"Errore: database utente non trovato: {db_path}")
            # Inizializza il database se non esiste
            init_success = init_user_db(session['username'])
            if not init_success:
                return jsonify({'error': 'Impossibile inizializzare il database utente. Prova a riparare il database.'}), 500
        
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Debug: stampa il percorso del database e il topic_id
        print(f"Cercando topic_id: {topic_id} nel database: {db_path}")
        
        # Verifica che la tabella learning_units esista
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_units'")
        if not c.fetchone():
            print("Errore: tabella learning_units non trovata")
            return jsonify({'error': 'Tabella delle lezioni non trovata. Prova a riparare il database.'}), 404
        
        # Recupera la lezione
        c.execute('''
            SELECT id, title, content, quiz, answers, total_questions, correct_answers 
            FROM learning_units 
            WHERE id = ?
        ''', (topic_id,))
        
        result = c.fetchone()
        if not result:
            print(f"Lezione non trovata con ID: {topic_id}")
            # Debug: stampa tutte le lezioni disponibili
            c.execute("SELECT id, title FROM learning_units")
            all_topics = c.fetchall()
            print(f"Lezioni disponibili: {all_topics}")
            
            # Se non ci sono lezioni, suggerisci di crearne una nuova
            if not all_topics:
                return jsonify({
                    'error': f'Nessuna lezione trovata. Crea una nuova lezione.',
                    'no_lessons': True
                }), 404
            
            # Suggerisci altre lezioni disponibili
            available_topics = [{'id': t[0], 'title': t[1]} for t in all_topics]
            return jsonify({
                'error': f'Lezione non trovata con ID: {topic_id}',
                'available_topics': available_topics
            }), 404
            
        # Converti i dati in un formato JSON
        topic = {
            'id': result[0],
            'title': result[1],
            'content': result[2],
            'quiz': json.loads(result[3]) if result[3] else [],
            'progress': {
                'total': result[5],
                'correct': result[6],
                'answers': json.loads(result[4]) if result[4] else {}
            }
        }
        
        conn.close()
        return jsonify(topic)
        
    except Exception as e:
        print(f"Errore nel recupero della lezione: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Errore nel recupero della lezione: {str(e)}',
            'suggestion': 'Prova a riparare il database o a creare una nuova lezione.'
        }), 500

@learning.route('/generate_content', methods=['POST'])
@login_required
def generate_content():
    try:
        data = request.get_json()
        print(f"Dati ricevuti: {data}")
        topic = data.get('title')
        if not topic:
            return jsonify({'error': 'Topic non specificato'}), 400

        print(f"Generazione lezione per: {topic}")

        # Genera il contenuto della lezione
        lesson_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Sei un professore esperto di Machine Learning. 
                Genera una lezione completa sull'argomento richiesto con la seguente struttura:
                
                1. Un titolo principale (# Titolo)
                2. Un breve abstract/introduzione di 2-3 frasi
                3. Diversi capitoli numerati con sottotitoli (## 1. Primo Capitolo)
                4. Ogni capitolo deve contenere spiegazioni dettagliate ed esempi
                5. Usa il markdown per formattare il testo
                
                La lezione deve essere completa ma concisa (circa 1000-1500 parole)."""},
                {"role": "user", "content": f"Crea una lezione su: {topic}"}
            ]
        )
        
        content = lesson_response.choices[0].message.content
        print(f"Contenuto lezione generato: {content[:100]}...")
        
        # Genera le domande del quiz
        quiz_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Genera 5 domande a risposta multipla sulla lezione. 
                Ogni domanda deve avere esattamente 4 opzioni e una sola risposta corretta.
                Formatta l'output come un array di oggetti JSON con questa struttura:
                [
                    {
                        "text": "testo della domanda",
                        "options": ["opzione 1", "opzione 2", "opzione 3", "opzione 4"],
                        "correct_answer": 0  // indice della risposta corretta (0-3)
                    }
                ]
                
                IMPORTANTE: Assicurati che l'output sia un JSON valido. Non includere testo prima o dopo il JSON.
                Usa solo la struttura JSON specificata sopra."""},
                {"role": "user", "content": f"Genera domande per questa lezione:\n\n{content}"}
            ]
        )
        
        quiz_content = quiz_response.choices[0].message.content
        print(f"Quiz generato (raw): {quiz_content[:100]}...")
        
        # Estrai e valida il JSON delle domande
        try:
            # Cerca di estrarre solo la parte JSON dal testo
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', quiz_content, re.DOTALL)
            if json_match:
                quiz_json = json_match.group(0)
                print(f"JSON estratto: {quiz_json[:100]}...")
                questions = json.loads(quiz_json)
            else:
                # Prova a caricare direttamente
                questions = json.loads(quiz_content)
            
            print(f"Domande caricate: {len(questions)}")
            
            # Valida che ogni domanda abbia il formato corretto
            for q in questions:
                if not isinstance(q, dict) or \
                   'text' not in q or \
                   'options' not in q or \
                   'correct_answer' not in q or \
                   len(q['options']) != 4 or \
                   not isinstance(q['correct_answer'], int) or \
                   q['correct_answer'] < 0 or \
                   q['correct_answer'] > 3:
                    raise ValueError(f"Formato domanda non valido: {q}")
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Errore nel parsing delle domande: {e}")
            print(f"Contenuto quiz completo: {quiz_content}")
            # Crea domande di fallback
            questions = [
                {
                    "text": "Domanda di esempio (errore nella generazione)",
                    "options": ["Opzione 1", "Opzione 2", "Opzione 3", "Opzione 4"],
                    "correct_answer": 0
                }
            ]
        
        # Salva nel database
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Debug: verifica la struttura della tabella
        try:
            c.execute("PRAGMA table_info(learning_units)")
            columns = c.fetchall()
            print(f"Struttura tabella learning_units: {columns}")
        except Exception as e:
            print(f"Errore nel verificare la struttura della tabella: {e}")
        
        try:
            c.execute('''
                INSERT INTO learning_units (title, content, quiz, answers, total_questions, correct_answers)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (topic, content, json.dumps(questions), json.dumps({}), len(questions), 0))
            
            unit_id = c.lastrowid
            print(f"Lezione salvata con ID: {unit_id}")
            conn.commit()
            
            # Verifica che la lezione sia stata salvata correttamente
            c.execute("SELECT id, title FROM learning_units WHERE id = ?", (unit_id,))
            saved_lesson = c.fetchone()
            print(f"Verifica salvataggio lezione: {saved_lesson}")
            
            conn.close()
            
            return jsonify({
                'id': unit_id,
                'title': topic,
                'content': content,
                'quiz': questions
            })
        except Exception as db_error:
            print(f"Errore nel salvare la lezione nel database: {db_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Errore nel salvare la lezione: {str(db_error)}'}), 500
        
    except Exception as e:
        print(f"Errore generale nella generazione del contenuto: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@learning.route('/submit_answer', methods=['POST'])
@login_required
def submit_answer():
    try:
        data = request.get_json()
        topic_id = data.get('topic_id')
        question_index = data.get('question_index')
        answer_index = data.get('answer_index')
        
        if not all([topic_id, isinstance(question_index, int), isinstance(answer_index, int)]):
            return jsonify({'error': 'Dati mancanti o non validi'}), 400
        
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Recupera la lezione e il quiz
        c.execute('SELECT quiz, answers, correct_answers FROM learning_units WHERE id = ?', (topic_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': 'Lezione non trovata'}), 404
            
        quiz = json.loads(result[0])
        answers = json.loads(result[1])
        correct_answers = result[2]
        
        # Verifica che l'indice della domanda sia valido
        if question_index < 0 or question_index >= len(quiz):
            return jsonify({'error': 'Indice domanda non valido'}), 400
            
        # Verifica se la risposta è corretta
        question = quiz[question_index]
        is_correct = answer_index == question['correct_answer']
        
        # Aggiorna le risposte e il conteggio
        answers[str(question_index)] = answer_index
        if is_correct:
            correct_answers += 1
            
        # Salva le modifiche
        c.execute('''
            UPDATE learning_units 
            SET answers = ?, correct_answers = ? 
            WHERE id = ?
        ''', (json.dumps(answers), correct_answers, topic_id))
        
        conn.commit()
        
        # Recupera i dati aggiornati per la risposta
        c.execute('''
            SELECT quiz, answers, total_questions, correct_answers 
            FROM learning_units 
            WHERE id = ?
        ''', (topic_id,))
        
        updated = c.fetchone()
        progress = {
            'total': updated[2],
            'correct': updated[3],
            'answers': json.loads(updated[1])
        }
        
        conn.close()
        
        return jsonify({
            'quiz': json.loads(updated[0]),
            'progress': progress
        })
        
    except Exception as e:
        print(f"Errore nel submit della risposta: {e}")
        return jsonify({'error': str(e)}), 500

@learning.route('/delete_lesson/<lesson_id>', methods=['DELETE'])
@login_required
def delete_lesson(lesson_id):
    try:
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Verifica che la lezione esista
        c.execute('SELECT id FROM topics WHERE id = ?', (lesson_id,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'Lezione non trovata'}), 404
        
        # Elimina la lezione
        c.execute('DELETE FROM topics WHERE id = ?', (lesson_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    except Exception as e:
        import traceback
        print('Errore in delete_lesson:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@learning.route('/delete_all_lessons', methods=['DELETE'])
@login_required
def delete_all_lessons():
    try:
        conn = get_user_db(session['username'])
        c = conn.cursor()
        
        # Elimina tutte le lezioni dell'utente
        c.execute('DELETE FROM topics')
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    
    except Exception as e:
        import traceback
        print('Errore in delete_all_lessons:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        print(f"Errore nell'eliminazione della lezione: {e}")
        return jsonify({'error': str(e)}), 500

@learning.route('/reset_db', methods=['POST'])
@login_required
def reset_db():
    try:
        username = session['username']
        
        # Ottieni il percorso del database
        db_path = get_user_db_path(username)
        
        # Chiudi eventuali connessioni aperte
        conn = get_user_db(username)
        conn.close()
        
        # Elimina il file del database se esiste
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Database eliminato: {db_path}")
        
        # Ricrea il database
        init_user_db(username)
        print(f"Database ricreato per l'utente: {username}")
        
        return jsonify({'success': True, 'message': 'Database reinizializzato con successo'})
        
    except Exception as e:
        print(f"Errore nel reset del database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@learning.route('/repair_db', methods=['POST'])
@login_required
def repair_db():
    """Verifica e ripara il database dell'utente."""
    try:
        username = session['username']
        print(f"Tentativo di riparazione del database per l'utente: {username}")
        
        # Ottieni il percorso del database
        db_path = get_user_db_path(username)
        print(f"Percorso database: {db_path}")
        
        # Verifica se il file esiste
        import os
        if not os.path.exists(db_path):
            print(f"Il database non esiste, verrà creato: {db_path}")
            # Il database verrà creato automaticamente quando ci connettiamo
        
        # Connetti al database
        conn = get_user_db(username)
        c = conn.cursor()
        
        # Verifica la struttura della tabella
        c.execute("PRAGMA table_info(learning_units)")
        columns = c.fetchall()
        print(f"Struttura tabella learning_units: {columns}")
        
        # Se la tabella non esiste, creala
        if not columns:
            print("La tabella learning_units non esiste, creazione in corso...")
            c.execute('''
            CREATE TABLE IF NOT EXISTS learning_units (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                quiz TEXT,
                answers TEXT DEFAULT '{}',
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            conn.commit()
            print("Tabella learning_units creata con successo")
        
        # Verifica l'integrità del database
        c.execute("PRAGMA integrity_check")
        integrity = c.fetchone()
        print(f"Controllo integrità: {integrity}")
        
        # Verifica se ci sono lezioni
        c.execute("SELECT COUNT(*) FROM learning_units")
        count = c.fetchone()[0]
        print(f"Numero di lezioni nel database: {count}")
        
        # Elenca tutte le lezioni
        c.execute("SELECT id, title FROM learning_units")
        lessons = c.fetchall()
        lesson_list = [{'id': row[0], 'title': row[1]} for row in lessons]
        print(f"Lezioni trovate: {lesson_list}")
        
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Database verificato e riparato con successo',
            'lessons_count': count,
            'lessons': lesson_list
        })
        
    except Exception as e:
        print(f"Errore durante la riparazione del database: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
