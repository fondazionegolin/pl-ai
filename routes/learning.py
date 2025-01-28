from flask import Blueprint, render_template, request, jsonify
import json
import re
import os
from markdown import markdown
from bs4 import BeautifulSoup
import uuid
from datetime import datetime
from .db import get_db
from .openai_client import client

learning = Blueprint('learning', __name__)

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS learning_units
                 (id TEXT PRIMARY KEY, 
                  title TEXT,
                  content TEXT,
                  quiz TEXT,
                  answers TEXT DEFAULT '[]',
                  total_questions INTEGER DEFAULT 0,
                  correct_answers INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

@learning.route('/learning')
def learning_page():
    init_db()  # Assicurati che il database sia inizializzato
    return render_template('learning.html')

def clean_and_format_markdown(content):
    """Pulisce e formatta il contenuto markdown per una presentazione migliore."""
    
    # Prima pulizia del contenuto
    content = content.strip()
    
    # Rimuove riferimenti numerici ai capitoli/sezioni
    content = re.sub(r'#\s*\d+[\.\)]\s*', '# ', content)
    content = re.sub(r'^\d+[\.\)]\s*', '', content, flags=re.MULTILINE)
    
    # Rimuove linee vuote multiple ma mantiene la spaziatura tra sezioni
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    # Normalizza i titoli e aggiunge spaziatura
    content = re.sub(r'\n\s*#{1,6}\s*(.+?)\s*#*$', r'\n\n# \1\n', content, flags=re.MULTILINE)
    content = re.sub(r'^#{1,6}\s*(.+?)\s*#*$', r'# \1\n', content, flags=re.MULTILINE)
    
    # Pulisce le liste e aggiunge spaziatura
    content = re.sub(r'\n\s*[-*+]\s+', '\n- ', content)
    content = re.sub(r'\n- ', '\n\n- ', content)
    content = re.sub(r'\n\s*\d+\.\s+', '\n1. ', content)
    content = re.sub(r'\n1. ', '\n\n1. ', content)
    
    # Pulisce il grassetto e il corsivo
    content = re.sub(r'\*\*\s*(.+?)\s*\*\*', r'**\1**', content)
    content = re.sub(r'\*\s*(.+?)\s*\*', r'*\1*', content)
    
    # Pulisce i blocchi di codice e aggiunge spaziatura
    content = re.sub(r'```\s*(\w+)?\s*\n', r'\n\n```\1\n', content)
    content = re.sub(r'\n\s*```\s*', r'\n```\n\n', content)
    
    # Pulisce le formule matematiche
    content = re.sub(r'\$\$\s*(.+?)\s*\$\$', r'\n\n$$\1$$\n\n', content, flags=re.DOTALL)
    content = re.sub(r'\$\s*(.+?)\s*\$', r'$\1$', content)
    
    # Converte il markdown in HTML
    html = markdown(
        content,
        extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists']
    )
    
    # Usa BeautifulSoup per migliorare la formattazione HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Rimuove classi esistenti
    for tag in soup.find_all(True):
        if 'class' in tag.attrs:
            del tag['class']
    
    # Aggiunge classi Tailwind per lo stile
    for tag in soup.find_all(['h1', 'h2', 'h3']):
        tag['class'] = 'text-2xl font-bold text-gray-800 mb-6 mt-8'
    
    # Formatta paragrafi con più spaziatura
    for p in soup.find_all('p'):
        if not p.get_text().strip():
            p.decompose()
            continue
        p['class'] = 'text-gray-700 mb-6 leading-relaxed max-w-3xl'
    
    # Formatta liste con più spaziatura
    for ul in soup.find_all('ul'):
        ul['class'] = 'list-disc pl-6 mb-8 space-y-4'
    for ol in soup.find_all('ol'):
        ol['class'] = 'list-decimal pl-6 mb-8 space-y-4'
    for li in soup.find_all('li'):
        li['class'] = 'text-gray-700 leading-relaxed'
    
    # Formatta blocchi di codice
    for pre in soup.find_all('pre'):
        pre['class'] = 'bg-gray-100 p-6 rounded-lg mb-8 overflow-x-auto'
    for code in soup.find_all('code'):
        if code.parent.name != 'pre':
            code['class'] = 'bg-gray-100 px-2 py-1 rounded text-sm font-mono'
        else:
            code['class'] = 'font-mono text-sm leading-relaxed'
    
    # Formatta tabelle con più spaziatura
    for table in soup.find_all('table'):
        table['class'] = 'min-w-full border border-gray-300 mb-8'
        wrapper = soup.new_tag('div')
        wrapper['class'] = 'overflow-x-auto mb-8'
        table.wrap(wrapper)
    
    for th in soup.find_all('th'):
        th['class'] = 'bg-gray-100 border border-gray-300 px-6 py-3 font-semibold'
    for td in soup.find_all('td'):
        td['class'] = 'border border-gray-300 px-6 py-3'
    
    # Formatta citazioni
    for blockquote in soup.find_all('blockquote'):
        blockquote['class'] = 'border-l-4 border-gray-300 pl-6 py-3 mb-8 text-gray-600 italic'
    
    # Rimuove spazi vuoti eccessivi
    html = str(soup)
    html = re.sub(r'\n\s*\n', '\n', html)
    html = re.sub(r'>\s+<', '><', html)
    
    return html

@learning.route('/get_topics', methods=['GET'])
def get_topics():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT id, title, answers, total_questions, correct_answers FROM learning_units')
        results = c.fetchall()
        conn.close()
        
        topics = []
        for result in results:
            answers = json.loads(result[2] if result[2] else '[]')
            total = result[3] or 0
            correct = result[4] or 0
            
            # Calcola lo stato
            status = 'not_started'  # rosso
            if total > 0:
                if correct == total:
                    status = 'completed'  # verde
                else:
                    status = 'in_progress'  # arancione
            
            topics.append({
                'id': result[0],
                'title': result[1],
                'status': status,
                'progress': {
                    'total': total,
                    'correct': correct,
                    'answers': answers
                }
            })
        
        return jsonify({'topics': topics})
    except Exception as e:
        print(f"Error in get_topics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@learning.route('/get_topic/<topic_id>', methods=['GET'])
def get_topic(topic_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT * FROM learning_units WHERE id = ?', (topic_id,))
        result = c.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Topic non trovato'}), 404
            
        print(f"Raw quiz data from DB: {result[3]}")  # Debug log
        
        try:
            # Assicuriamoci che il quiz sia un array valido
            quiz_data = json.loads(result[3]) if result[3] else []
            if not isinstance(quiz_data, list):
                print(f"Quiz data is not a list: {type(quiz_data)}")  # Debug log
                quiz_data = []
                
            # Validiamo la struttura di ogni domanda
            valid_quiz = []
            for q in quiz_data:
                if isinstance(q, dict) and 'question' in q and 'options' in q:
                    valid_quiz.append(q)
                else:
                    print(f"Invalid question format: {q}")  # Debug log
            
            # Decodifica le risposte con gestione errori
            try:
                answers = json.loads(result[4]) if result[4] else []
            except json.JSONDecodeError:
                print(f"Invalid answers data: {result[4]}")  # Debug log
                answers = []
            
            total_questions = result[5] or 0
            correct_answers = result[6] or 0
            
            response_data = {
                'id': result[0],
                'title': result[1],
                'content': result[2],
                'quiz': valid_quiz,
                'progress': {
                    'total': total_questions,
                    'correct': correct_answers,
                    'answers': answers
                }
            }
            
            print(f"Sending response: {json.dumps(response_data)}")  # Debug log
            return jsonify(response_data)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing quiz data: {str(e)}")
            print(f"Quiz data type: {type(result[3])}")
            print(f"Quiz data: {result[3]}")
            return jsonify({'error': 'Invalid quiz data'}), 500
                
    except Exception as e:
        print(f"Error in get_topic: {str(e)}")
        return jsonify({'error': str(e)}), 500

@learning.route('/generate_content', methods=['POST'])
def generate_content():
    try:
        data = request.json
        topic = data.get('topic', '')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        # Genera il contenuto
        content_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Sei un professore esperto di Machine Learning.
                Crea una lezione chiara e concisa sull'argomento richiesto, seguendo queste linee guida:
                
                STRUTTURA:
                - Breve introduzione accattivante
                - Spiegazione dei concetti chiave
                - Esempi pratici e analogie
                - Breve riassunto
                
                CONTENUTO:
                - Un concetto per paragrafo
                - Frasi brevi e dirette
                - Esempi concreti
                - Formule solo se essenziali
                
                FORMATTAZIONE:
                - Solo titoli di primo livello (#)
                - Vai a capo spesso
                - Usa liste puntate
                - Mantieni tutto pulito e semplice"""},
                {"role": "user", "content": f"Crea una lezione su: {topic}"}
            ]
        )
        
        content = content_response.choices[0].message.content
        
        # Genera il quiz
        quiz_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Crea 5 domande a risposta multipla sulla lezione.
                Restituisci le domande in questo formato JSON (e SOLO questo, niente testo prima o dopo):
                [
                    {
                        "question": "Domanda 1?",
                        "options": [
                            {"text": "Risposta 1", "correct": true},
                            {"text": "Risposta 2", "correct": false},
                            {"text": "Risposta 3", "correct": false},
                            {"text": "Risposta 4", "correct": false}
                        ]
                    }
                ]"""},
                {"role": "user", "content": f"Crea un quiz sulla lezione appena generata su {topic}"}
            ]
        )
        
        quiz_text = quiz_response.choices[0].message.content.strip()
        print(f"Raw quiz response: {quiz_text}")  # Debug log
        
        try:
            quiz = json.loads(quiz_text)
            if not isinstance(quiz, list):
                print(f"Quiz is not a list: {type(quiz)}")  # Debug log
                raise ValueError("Quiz data is not an array")
                
            # Valida la struttura di ogni domanda
            for q in quiz:
                if not isinstance(q, dict) or 'question' not in q or 'options' not in q:
                    print(f"Invalid question format: {q}")  # Debug log
                    raise ValueError("Invalid question format")
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing quiz JSON: {str(e)}")
            print(f"Quiz text: {quiz_text}")
            return jsonify({'error': 'Failed to generate quiz'}), 500
        
        # Salva nel database
        unit_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        content_html = clean_and_format_markdown(content)
        quiz_json = json.dumps(quiz)
        
        print(f"Saving quiz to DB: {quiz_json}")  # Debug log
        
        conn = get_db()
        c = conn.cursor()
        c.execute('''INSERT INTO learning_units (id, title, content, quiz)
                    VALUES (?, ?, ?, ?)''',
                 (unit_id, topic, content_html, quiz_json))
        conn.commit()
        conn.close()
        
        response_data = {
            'id': unit_id,
            'title': topic,
            'content': content_html,
            'quiz': quiz,
            'progress': {
                'total': 0,
                'correct': 0,
                'answers': []
            }
        }
        
        print(f"Sending response: {json.dumps(response_data)}")  # Debug log
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in generate_content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@learning.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        data = request.json
        unit_id = data.get('unit_id')
        question_index = data.get('question_index')
        answer = data.get('answer')
        is_correct = data.get('is_correct')

        conn = get_db()
        c = conn.cursor()
        
        # Ottieni le risposte esistenti
        c.execute('SELECT answers, total_questions, correct_answers FROM learning_units WHERE id = ?', (unit_id,))
        result = c.fetchone()
        if not result:
            return jsonify({'error': 'Unit not found'}), 404
            
        answers = json.loads(result[0] if result[0] else '[]')
        total_questions = result[1] or 0
        correct_answers = result[2] or 0
        
        # Aggiorna o aggiungi la nuova risposta
        while len(answers) <= question_index:
            answers.append(None)
        
        if answers[question_index] is None:
            total_questions += 1
            if is_correct:
                correct_answers += 1
        elif answers[question_index] != is_correct:  # Se la risposta è cambiata
            if is_correct:
                correct_answers += 1
            else:
                correct_answers -= 1
                
        answers[question_index] = is_correct
        
        # Salva le risposte aggiornate
        c.execute('''UPDATE learning_units 
                    SET answers = ?, total_questions = ?, correct_answers = ? 
                    WHERE id = ?''', 
                 (json.dumps(answers), total_questions, correct_answers, unit_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'progress': {
                'total': total_questions,
                'correct': correct_answers,
                'answers': answers
            }
        })
    except Exception as e:
        print(f"Error in submit_answer: {str(e)}")
        return jsonify({'error': str(e)}), 500
