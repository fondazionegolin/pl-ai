from flask import Blueprint, render_template, request, jsonify, session
from .db import login_required
from .openai_client import client
import json
import re

mathematics = Blueprint('mathematics', __name__)

@mathematics.route('/matematica')
@login_required
def matematica():
    return render_template('matematica.html')

@mathematics.route('/api/generate-lesson', methods=['POST'])
@login_required
def generate_lesson():
    try:
        data = request.get_json()
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        
        prompt = f"""Genera una mini-lezione di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}

La lezione deve:
1. Essere concisa (max 15 righe)
2. Spiegare i concetti in modo chiaro e didattico
3. Includere esempi pratici
4. Usare formule in LaTeX quando necessario (es: \\(x^2 + 3x = 0\\))
5. Essere adatta al livello scolastico indicato"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un insegnante di matematica esperto. Scrivi lezioni chiare e didattiche."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        lesson = response.choices[0].message.content.strip()
        
        return jsonify({
            'status': 'success',
            'lesson': lesson
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@mathematics.route('/api/generate-exercise', methods=['POST'])
@login_required
def generate_exercise():
    try:
        data = request.get_json()
        grade = data.get('grade')
        subject = data.get('subject')
        topic = data.get('topic')
        exercise_type = data.get('exerciseType')
        
        prompt = f"""Genera un esercizio di matematica per:
- Classe: {grade}
- Materia: {subject}
- Argomento: {topic}
- Tipo: {exercise_type}

Formato richiesto:
1. Domanda chiara in italiano
2. Formule in LaTeX (es: \\(x^2 + 3x = 0\\))
3. Risposta corretta tra [RISPOSTA]...[/RISPOSTA]

Per il tipo di esercizio:
- 'fill': domanda con risposta numerica o algebrica
- 'text': domanda che richiede una spiegazione in italiano
- 'image': domanda che richiede di mostrare i passaggi della soluzione"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un tutor di matematica esperto. Genera esercizi appropriati per il livello scolastico indicato."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        exercise = content.split('[RISPOSTA]')[0].strip()
        answer = content.split('[RISPOSTA]')[1].split('[/RISPOSTA]')[0].strip()
        
        return jsonify({
            'status': 'success',
            'exercise': exercise,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@mathematics.route('/api/check-answer', methods=['POST'])
@login_required
def check_answer():
    try:
        data = request.get_json()
        answer = data.get('answer')
        exercise_type = data.get('exerciseType')
        correct_answer = data.get('correctAnswer')
        
        if exercise_type == 'fill':
            # Per risposte numeriche/algebriche, normalizza e confronta
            is_correct = normalize_answer(answer) == normalize_answer(correct_answer)
            feedback = "Risposta corretta!" if is_correct else f"Risposta errata. La soluzione corretta è: {correct_answer}"
        
        elif exercise_type == 'text':
            # Per risposte testuali, usa GPT per valutare
            prompt = f"""Valuta se la risposta dello studente è corretta rispetto alla soluzione attesa.
            
Soluzione attesa:
{correct_answer}

Risposta dello studente:
{answer}

Rispondi con un JSON nel formato:
{{
    "is_correct": true/false,
    "feedback": "spiegazione dettagliata"
}}"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un insegnante di matematica che valuta le risposte degli studenti."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            is_correct = result['is_correct']
            feedback = result['feedback']
        
        else:
            return jsonify({'status': 'error', 'message': 'Tipo di esercizio non supportato'}), 400
        
        return jsonify({
            'status': 'success',
            'isCorrect': is_correct,
            'feedback': feedback
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def normalize_answer(answer):
    """Normalizza una risposta matematica per il confronto"""
    if not answer:
        return ""
    # Rimuovi spazi extra e converti in minuscolo
    answer = answer.strip().lower()
    # Sostituisci caratteri speciali
    answer = answer.replace('×', '*').replace('÷', '/')
    # Rimuovi spazi tra numeri e operatori
    answer = re.sub(r'(\d)\s+([+\-*/])\s+(\d)', r'\1\2\3', answer)
    return answer 