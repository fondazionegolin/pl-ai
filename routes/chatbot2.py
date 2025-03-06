from flask import Blueprint, render_template, request, jsonify
import sqlite3
import json
import os
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Initialize API clients and validate API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Dictionary to track API client initialization status
api_status = {
    'openai': False,
    'anthropic': False,
    'gemini': False
}

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
        anthropic_client = Anthropic(api_key=anthropic_api_key)
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
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot2_chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.commit()
    conn.close()

# Initialize database tables
init_db()

@chatbot2.route('/chatbot2')
def chat_page():
    return render_template('chatbot2.html')

@chatbot2.route('/api/chat-history')
def get_chat_history():
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM chatbot2_chats ORDER BY created_at DESC')
    chats = [{'id': row[0], 'title': row[1], 'created_at': row[2]} for row in c.fetchall()]
    conn.close()
    return jsonify(chats)

@chatbot2.route('/api/chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    conn = None
    try:
        conn = sqlite3.connect('database.sqlite')
        c = conn.cursor()
        
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

@chatbot2.route('/api/chat/<int:chat_id>')
def get_chat(chat_id):
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    
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
                c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 10', (chat_id,))
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
                
            elif model_name == 'claude':
                if not anthropic_api_key:
                    raise Exception("Anthropic API key not configured. Please check your environment variables.")

                try:
                    # Get chat history for context
                    c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 10', (chat_id,))
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
                            message_response = anthropic_client.messages.create(
                                model=model,
                                max_tokens=4096,
                                messages=messages,
                                temperature=0.7,
                                system=system_prompt if system_prompt else None
                            )

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
                c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 10', (chat_id,))
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
            
        # For interrogation mode, ensure the bot always asks a question
        if context.get('mode') == 'interrogazione' and message != 'START_INTERROGATION':
            # Get previous messages to maintain context
            c.execute('SELECT role, content FROM chatbot2_messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT 10', (chat_id,))
            chat_history = c.fetchall()
            
            # Add a reminder to evaluate and ask next question
            if model_name == 'gpt4':
                completion = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[{"role": msg[0], "content": msg[1]} for msg in reversed(chat_history)],
                        {"role": "system", "content": "Valuta la risposta dell'utente, fornisci un feedback costruttivo e poi fai una nuova domanda pertinente."}
                    ]
                )
                response = completion.choices[0].message.content
            elif model_name == 'claude':
                messages = [
                    {"role": "system", "content": system_prompt},
                    *[{"role": msg[0], "content": msg[1]} for msg in reversed(chat_history)],
                    {"role": "system", "content": "Valuta la risposta dell'utente, fornisci un feedback costruttivo e poi fai una nuova domanda pertinente."}
                ]
                message_response = anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=messages
                )
                response = message_response.content[0].text
            else:  # gemini
                # Create model
                model = genai.GenerativeModel('gemini-1.5-pro-001')
                
                # Prepare context
                history_text = "\n".join([f"{msg[0]}: {msg[1]}" for msg in reversed(chat_history)])
                prompt = f"{system_prompt}\n\nStorico della conversazione:\n{history_text}\n\nValuta la risposta dell'utente, fornisci un feedback costruttivo e poi fai una nuova domanda pertinente."
                
                # Generate response
                result = model.generate_content(prompt)
                response = result.text
            
        # Save AI response
        c.execute('''INSERT INTO chatbot2_messages (chat_id, role, content, model)
                     VALUES (?, ?, ?, ?)''',
                 (chat_id, 'assistant', response, context.get('model')))
        
        conn.commit()
        
        return jsonify({
            'response': response,
            'chatId': chat_id
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        if conn:
            conn.rollback()
            return jsonify({'error': str(e)}), 500
            
    finally:
        if conn:
            conn.close()
