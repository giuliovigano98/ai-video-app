# ===============================
# Stage 1 — Build environment
# ===============================
FROM python:3.11-slim AS build

WORKDIR /app

# Install build tools (necessari per TensorFlow e OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
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

# Install ffmpeg and minimal libs for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copia env Python da build stage
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /usr/local/bin /usr/local/bin

# Copia codice app
COPY . .

# Espone porta Streamlit
EXPOSE 8501

# Avvio Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
