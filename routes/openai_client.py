import os
from openai import OpenAI

# Inizializza il client OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
