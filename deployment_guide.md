# Guida al Deployment di PL-AI su VPS Debian

Questa guida ti mostrerà come deployare l'applicazione PL-AI su una VPS Debian, rendendola accessibile all'URL `dominiovps/pl-ai`.

## 1. Preparazione della VPS

### Aggiornare il sistema
```bash
sudo apt update
sudo apt upgrade -y
```

### Installare le dipendenze necessarie
```bash
sudo apt install -y python3 python3-pip python3-venv nginx supervisor git
```

## 2. Clonare il repository

### Creare una directory per l'applicazione
```bash
sudo mkdir -p /var/www/pl-ai
sudo chown $USER:$USER /var/www/pl-ai
```

### Clonare il repository
```bash
cd /var/www/pl-ai
git clone https://github.com/fondazionegolin/pl-ai.git .
```

## 3. Configurare l'ambiente Python

### Creare un ambiente virtuale
```bash
python3 -m venv venv
source venv/bin/activate
```

### Installare le dipendenze
```bash
pip install -r requirements.txt
pip install gunicorn
```

## 4. Configurare i log

### Creare directory per i log
```bash
sudo mkdir -p /var/log/pl-ai
sudo chown $USER:$USER /var/log/pl-ai
```

## 5. Configurare Supervisor

Supervisor è uno strumento che gestirà il processo Gunicorn e lo riavvierà automaticamente in caso di crash.

### Creare un file di configurazione per Supervisor
```bash
sudo nano /etc/supervisor/conf.d/pl-ai.conf
```

Inserisci il seguente contenuto:
```ini
[program:pl-ai]
directory=/var/www/pl-ai
command=/var/www/pl-ai/venv/bin/gunicorn --config gunicorn_config.py wsgi:app
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/pl-ai/supervisor.err.log
stdout_logfile=/var/log/pl-ai/supervisor.out.log
environment=PATH="/var/www/pl-ai/venv/bin"
```

### Aggiornare Supervisor
```bash
sudo chown -R www-data:www-data /var/www/pl-ai
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start pl-ai
```

## 6. Configurare Nginx

Nginx fungerà da reverse proxy, inoltrando le richieste a Gunicorn.

### Creare un file di configurazione per Nginx
```bash
sudo nano /etc/nginx/sites-available/pl-ai
```

Inserisci il seguente contenuto (sostituisci `dominiovps` con il tuo dominio effettivo):
```nginx
server {
    listen 80;
    server_name dominiovps;

    location /pl-ai {
        rewrite ^/pl-ai(/.*)$ $1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /pl-ai/static {
        alias /var/www/pl-ai/static;
    }

    location /pl-ai/uploads {
        alias /var/www/pl-ai/uploads;
    }
}
```

### Attivare la configurazione di Nginx
```bash
sudo ln -s /etc/nginx/sites-available/pl-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 7. Configurare il Firewall (opzionale)

Se hai un firewall attivo, assicurati che le porte 80 e 443 siano aperte:
```bash
sudo ufw allow 80
sudo ufw allow 443
```

## 8. Configurare HTTPS con Let's Encrypt (opzionale ma consigliato)

Per una maggiore sicurezza, è consigliabile configurare HTTPS:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d dominiovps
```

## 9. Verificare il Deployment

Ora dovresti essere in grado di accedere all'applicazione all'URL `http://dominiovps/pl-ai` (o `https://dominiovps/pl-ai` se hai configurato HTTPS).

## 10. Risoluzione dei problemi

### Controllare i log di Supervisor
```bash
sudo tail -f /var/log/pl-ai/supervisor.err.log
sudo tail -f /var/log/pl-ai/supervisor.out.log
```

### Controllare i log di Gunicorn
```bash
sudo tail -f /var/log/pl-ai/error.log
sudo tail -f /var/log/pl-ai/access.log
```

### Controllare i log di Nginx
```bash
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### Riavviare i servizi
```bash
sudo supervisorctl restart pl-ai
sudo systemctl restart nginx
```

## 11. Aggiornare l'applicazione

Quando vuoi aggiornare l'applicazione con nuove modifiche:
```bash
cd /var/www/pl-ai
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo supervisorctl restart pl-ai
```
