# Guida al Deployment di PL-AI su VPS

Questa guida fornisce istruzioni dettagliate per il deployment dell'applicazione PL-AI su una VPS utilizzando NGINX come web server e Gunicorn come server WSGI.

## Prerequisiti

- VPS con Ubuntu 20.04 o superiore
- Accesso SSH alla VPS
- Dominio pl-ai.it configurato per puntare all'IP della VPS
- Conoscenze di base di Linux, NGINX e Python

## 1. Preparazione dell'ambiente sulla VPS

```bash
# Aggiorna il sistema
sudo apt update && sudo apt upgrade -y

# Installa le dipendenze necessarie
sudo apt install -y python3-pip python3-dev python3-venv nginx certbot python3-certbot-nginx git

# Crea directory per i log
sudo mkdir -p /var/log/pl-ai
sudo chown -R ubuntu:ubuntu /var/log/pl-ai
```

## 2. Clonare il repository e configurare l'ambiente Python

```bash
# Clona il repository nella home directory
cd ~
git clone https://github.com/fondazionegolin/pl-ai.git
cd pl-ai

# Crea e attiva un ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
pip install gunicorn

# Crea le directory necessarie se non esistono
mkdir -p pictures
```

## 3. Configurare il file .env

Crea un file `.env` nella directory del progetto con le variabili d'ambiente necessarie:

```bash
touch .env
nano .env
```

Aggiungi le seguenti variabili (sostituisci con i tuoi valori):

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
SECRET_KEY=your_flask_secret_key
DATABASE_URL=your_database_url
```

## 4. Configurare NGINX

```bash
# Copia il file di configurazione NGINX
sudo cp ~/pl-ai/nginx/pl-ai.conf /etc/nginx/sites-available/pl-ai.conf

# Crea un link simbolico nella directory sites-enabled
sudo ln -s /etc/nginx/sites-available/pl-ai.conf /etc/nginx/sites-enabled/

# Verifica la configurazione di NGINX
sudo nginx -t

# Se la verifica è andata a buon fine, riavvia NGINX
sudo systemctl restart nginx
```

## 5. Configurare SSL con Let's Encrypt

```bash
# Ottieni un certificato SSL per il tuo dominio
sudo certbot --nginx -d pl-ai.it -d www.pl-ai.it

# Segui le istruzioni a schermo per completare la configurazione
```

## 6. Configurare il servizio systemd

```bash
# Copia il file di servizio
sudo cp ~/pl-ai/pl-ai.service /etc/systemd/system/

# Ricarica la configurazione di systemd
sudo systemctl daemon-reload

# Abilita il servizio per l'avvio automatico
sudo systemctl enable pl-ai

# Avvia il servizio
sudo systemctl start pl-ai

# Verifica lo stato del servizio
sudo systemctl status pl-ai
```

## 7. Monitoraggio e manutenzione

### Controllare i log

```bash
# Log di Gunicorn
sudo tail -f /var/log/pl-ai/error.log
sudo tail -f /var/log/pl-ai/access.log

# Log di NGINX
sudo tail -f /var/log/nginx/pl-ai.error.log
sudo tail -f /var/log/nginx/pl-ai.access.log
```

### Riavviare il servizio dopo aggiornamenti

```bash
# Pull delle nuove modifiche
cd ~/pl-ai
git pull

# Attiva l'ambiente virtuale e aggiorna le dipendenze se necessario
source venv/bin/activate
pip install -r requirements.txt

# Riavvia il servizio
sudo systemctl restart pl-ai
```

## 8. Risoluzione dei problemi comuni

### Problema: La pagina non viene aggiornata con le ultime modifiche

1. **Svuota la cache del browser**: Premi Ctrl+F5 o Cmd+Shift+R
2. **Riavvia Gunicorn**: `sudo systemctl restart pl-ai`
3. **Controlla i log per errori**: Vedi sezione "Controllare i log"
4. **Verifica i permessi dei file**: Assicurati che l'utente ubuntu abbia accesso in lettura a tutti i file

### Problema: Errori 502 Bad Gateway

1. **Verifica che Gunicorn sia in esecuzione**: `sudo systemctl status pl-ai`
2. **Controlla i log di Gunicorn e NGINX**
3. **Verifica la configurazione di NGINX**: `sudo nginx -t`

### Problema: Certificato SSL scaduto

```bash
# Rinnova il certificato SSL
sudo certbot renew
```

## 9. Backup e ripristino

### Backup del database e dei file caricati

```bash
# Backup del database SQLite
cp ~/pl-ai/instance/database.db ~/backups/database_$(date +%Y%m%d).db

# Backup delle immagini caricate
tar -czf ~/backups/pictures_$(date +%Y%m%d).tar.gz ~/pl-ai/pictures
```

## 10. Ottimizzazione delle prestazioni

Per migliorare le prestazioni dell'applicazione, puoi:

1. Aumentare il numero di worker Gunicorn nel file `gunicorn_config.py`
2. Configurare la cache di NGINX per i file statici
3. Utilizzare un CDN per servire i file statici
4. Implementare il caching a livello di applicazione

---

Per qualsiasi problema o domanda, consulta la documentazione ufficiale di [Flask](https://flask.palletsprojects.com/), [Gunicorn](https://docs.gunicorn.org/) e [NGINX](https://nginx.org/en/docs/).
