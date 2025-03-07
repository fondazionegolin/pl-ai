FROM python:3.12-slim

WORKDIR /app

# Installa le dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copia i file dei requisiti
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Installa gunicorn esplicitamente
RUN pip install --no-cache-dir gunicorn==21.2.0

# Copia il resto dell'applicazione
COPY . .

# Crea directory per i file caricati se non esiste
RUN mkdir -p uploads
RUN mkdir -p static/generated

# Imposta le autorizzazioni
RUN chmod -R 755 /app

# Espone la porta che l'app utilizzerà
EXPOSE 5002

# Comando per avviare l'applicazione
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]
