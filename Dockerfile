# ===============================
# Stage 1 — Build environment
# ===============================
FROM python:3.11-slim AS build

WORKDIR /app

# Pacchetti minimi per TF/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ===============================
# Stage 2 — Runtime
# ===============================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime libs per video
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Copia l'env Python dallo stage build (⚠️ 3.11, non 3.10)
COPY --from=build /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=build /usr/local/bin /usr/local/bin

# Copia codice app
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
