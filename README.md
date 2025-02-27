# PL-AI Platform

PL-AI è una piattaforma web per l'analisi dei dati e il machine learning, progettata per rendere accessibili strumenti avanzati di analisi attraverso un'interfaccia utente intuitiva.

## 🚀 Funzionalità

### 🤖 Intelligenza Artificiale Conversazionale
- **Chatbot Avanzato**: Assistente AI per supporto e analisi
  - Interfaccia conversazionale naturale
  - Analisi contestuale delle richieste
  - Supporto multilingua
  - Integrazione con le funzionalità della piattaforma
  - Personalizzazione del comportamento

### 📊 Analisi Dati
- **Classificazione**: Addestra modelli di classificazione su dataset CSV
  - Selezione automatica e manuale delle feature
  - Visualizzazione dei risultati del training
  - Interfaccia a step per una migliore esperienza utente
  - Predizioni in tempo reale

- **Regressione**: Analisi predittiva su dati numerici
  - Supporto per dataset CSV
  - Selezione interattiva di target e feature
  - Metriche di performance (R² score)
  - Interfaccia guidata step-by-step

### 🖼️ Computer Vision
- **Classificazione Immagini**: Riconoscimento e classificazione di immagini
  - Upload di immagini singole
  - Predizioni in tempo reale
  - Supporto per webcam
  - Visualizzazione delle probabilità per classe

- **Generazione Immagini**: Creazione di immagini con AI
  - Generazione basata su descrizione testuale
  - Modifica di immagini esistenti (inpainting)
  - Stili artistici personalizzabili
  - Controllo della composizione
  - Export in vari formati

### 📚 Learning Platform
- **Corsi Interattivi**: Piattaforma di apprendimento integrata
  - Tutorial guidati per ogni funzionalità
  - Esempi pratici e casi d'uso
  - Quiz e verifiche di apprendimento
  - Tracciamento dei progressi

- **Documentazione**: Risorse di apprendimento complete
  - Guide passo-passo
  - Best practices
  - FAQ e troubleshooting
  - Video tutorial

- **Progetti Esempio**: Raccolta di progetti dimostrativi
  - Dataset preconfigurati
  - Notebook interattivi
  - Scenari reali
  - Soluzioni commentate

## 🛠 Tecnologie Utilizzate

### Backend
- **Python**: Linguaggio principale per il backend
- **Flask**: Framework web
- **scikit-learn**: Libreria per machine learning
- **TensorFlow/Keras**: Per deep learning e computer vision
- **Transformers**: Per modelli linguistici e chatbot
- **Stable Diffusion**: Per generazione di immagini
- **pandas**: Per la manipolazione dei dati
- **numpy**: Per calcoli numerici
- **FastAPI**: Per API ad alte prestazioni
- **SQLAlchemy**: Per gestione database

### Frontend
- **HTML/CSS**: Struttura e stile dell'interfaccia
- **JavaScript**: Interattività lato client
- **Tailwind CSS**: Framework CSS per il design
- **Font Awesome**: Icone e elementi grafici

## 📦 Requisiti di Sistema
- Python 3.8+
- pip (Python package installer)
- Webcam (opzionale, per funzionalità di classificazione immagini in tempo reale)

## 🚀 Installazione

1. Clona il repository:
```bash
git clone https://github.com/fondazionegolin/pl-ai.git
cd pl-ai
```

2. Crea un ambiente virtuale (consigliato):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Avvia l'applicazione:
```bash
python app.py
```

L'applicazione sarà disponibile all'indirizzo `http://localhost:5000`

## 🔧 Configurazione

### Variabili d'Ambiente
Crea un file `.env` nella root del progetto:
```env
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your-secret-key
```

### Directory dei Dati
L'applicazione utilizza le seguenti directory per i dati:
- `/uploads`: File caricati dagli utenti
- `/models`: Modelli addestrati salvati
- `/temp`: File temporanei

## 📖 Utilizzo

### Classificazione CSV
1. Carica un file CSV
2. Seleziona la variabile target e le feature
3. Avvia il training
4. Effettua predizioni sui nuovi dati

### Regressione
1. Carica un file CSV
2. Seleziona la variabile target (numerica) e le feature
3. Avvia il training
4. Visualizza le metriche di performance
5. Effettua predizioni

### Classificazione Immagini
1. Carica un'immagine o utilizza la webcam
2. Visualizza l'anteprima
3. Ottieni le predizioni con le relative probabilità

### Generazione Immagini
1. Inserisci una descrizione testuale dettagliata
2. Seleziona lo stile artistico desiderato
3. Configura i parametri di generazione
4. Genera e scarica l'immagine
5. Modifica l'immagine se necessario

### Chatbot
1. Accedi all'interfaccia di chat
2. Poni domande o richiedi analisi
3. Ricevi risposte contestuali e suggerimenti
4. Utilizza i comandi speciali per funzionalità avanzate

### Learning Platform
1. Registrati alla piattaforma
2. Scegli un percorso di apprendimento
3. Completa i tutorial interattivi
4. Verifica la comprensione con i quiz
5. Applica le conoscenze ai progetti esempio

## 🤝 Contribuire

Siamo aperti a contributi! Per favore:
1. Fai un fork del repository
2. Crea un branch per le tue modifiche
3. Invia una pull request

## 📝 Licenza

Questo progetto è sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 👥 Team

- Fondazione Golin
- Maintainers e contributors

## 📞 Contatti

Per supporto o domande:
- Email: [contact@fondazionegolin.it](mailto:contact@fondazionegolin.it)
- Issues: [GitHub Issues](https://github.com/fondazionegolin/pl-ai/issues)
