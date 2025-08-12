# app/inference.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MOBILENET_PATH = MODELS_DIR / "mobilenet_tank_ifv_model.h5"
TANK_PATH      = MODELS_DIR / "threat_classifier_modelTANK.h5"
IFV_PATH       = MODELS_DIR / "threat_classifier_modelIFV.h5"

# mapping (adatta se i tuoi modelli hanno ordine diverso)
CLASS_MAP_MAIN = {0: "IFV", 1: "Tank"}
THREAT_LABELS  = ["Low", "Medium", "High"]  # ordine dell'output del modello

# --- Caricamento modelli (una sola volta) ---
mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
tank_model      = tf.keras.models.load_model(TANK_PATH)
ifv_model       = tf.keras.models.load_model(IFV_PATH)

def _preprocess_frame(frame, size=(224, 224)):
    img = cv2.resize(frame, size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def _softmax_to_percentages(logits_or_probs: np.ndarray):
    """
    Accetta vettore 1xC. Se non è normalizzato, applica softmax.
    Ritorna percentuali (somma 100).
    """
    v = logits_or_probs.astype(np.float32).ravel()
    # normalizza se necessario
    if np.any(v < 0) or not np.isclose(np.sum(v), 1.0, atol=1e-3):
        e = np.exp(v - np.max(v))
        v = e / np.sum(e)
    v = (v / np.sum(v)) * 100.0
    return v

def _draw_overlay_bottom_right(frame, equip_label, threat_perc):
    """
    Disegna in basso a destra:
      - testo equip_label
      - 3 barre High/Medium/Low con percentuali.
    threat_perc: array di 3 valori [Low, Medium, High] in percentuale.
    """
    h, w = frame.shape[:2]

    # box overlay
    pad = 16
    box_w = 220   # larghezza ridotta
    bar_h = 14    # barre più basse
    gap = 6
    header_h = 22
    total_h = header_h + gap + 3 * (bar_h + gap) + pad
    x2, y2 = w - pad, h - pad
    x1, y1 = x2 - box_w, y2 - total_h

    # sfondo traslucido
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # header (equip label)
    cv2.putText(frame, f"{equip_label}",
                (x1 + 12, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # barre
    labels = ["High", "Medium", "Low"]
    # threat_perc è [Low, Med, High] -> rimappo nello stesso ordine delle labels
    perc_map = {
        "Low":   float(threat_perc[0]),
        "Medium": float(threat_perc[1]),
        "High":   float(threat_perc[2]),
    }

    max_bar_w = box_w - 24 - 64  # spazio per etichetta a sinistra
    base_y = y1 + header_h + gap

    for i, lab in enumerate(labels):
        y = base_y + i * (bar_h + gap)
        # etichetta
        cv2.putText(frame, lab, (x1 + 12, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        # barra
        pct = max(0.0, min(100.0, perc_map[lab]))
        bar_w = int(max_bar_w * (pct / 100.0))
        bar_x = x1 + 12 + 64
        # contorno
        cv2.rectangle(frame, (bar_x, y), (bar_x + max_bar_w, y + bar_h), (220, 220, 220), 1)
        # riempimento
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (255, 255, 255), -1)
        # percentuale testo
        cv2.putText(frame, f"{pct:.0f}%", (bar_x + max_bar_w + 6, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    return frame

def _preprocess_for_model(model, frame):
    """
    Prepara il frame per *questo* modello:
    - legge model.input_shape per capire H, W e canali attesi
    - fa resize
    - converte a grayscale se serve 1 canale
    - normalizza in [0,1]
    - restituisce shape (1, H, W, C) oppure (1, C, H, W) se channels_first
    """
    in_shape = model.input_shape  # es. (None, 224, 224, 1) oppure (None, 224, 224, 3) o (None, 3, 224, 224)
    if isinstance(in_shape, list):  # alcuni modelli hanno più input
        in_shape = in_shape[0]

    # individua ordine canali
    if len(in_shape) == 4:
        b, d1, d2, d3 = in_shape
        # due casi più comuni:
        # channels_last: (None, H, W, C)
        # channels_first: (None, C, H, W)
        if d3 in (1, 3):
            fmt = "channels_last"
            H, W, C = d1, d2, d3
        elif d1 in (1, 3):
            fmt = "channels_first"
            C, H, W = d1, d2, d3
        else:
            # fallback: assumo channels_last
            fmt = "channels_last"
            H, W, C = d1, d2, d3
    else:
        # fallback ragionevole
        fmt = "channels_last"
        H, W, C = 224, 224, 3

    # resize
    img = cv2.resize(frame, (W, H))

    # gestisci canali
    if C == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H, W)
        img = np.expand_dims(img, axis=-1)          # (H, W, 1)
    else:
        # assicura BGR->RGB se il modello lo richiede; non lo sappiamo, quindi lasciamo BGR.
        pass

    img = img.astype(np.float32) / 255.0

    if fmt == "channels_first":
        # (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))

    return np.expand_dims(img, axis=0)  # aggiungi batch dim

def process_video(input_path: str, output_path: str, every_nth_frame: int = 1, smooth: int = 5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Impossibile aprire il video di input.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    buf_low, buf_med, buf_high = [], [], []
    i = 0
    last_equip_label = "..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % every_nth_frame == 0:
            # --- classificazione equipaggiamento (MobileNet) ---
            x_main = _preprocess_for_model(mobilenet_model, frame)
            main_probs = mobilenet_model.predict(x_main, verbose=0)
            main_idx = int(np.argmax(main_probs, axis=1)[0])
            equip_label = CLASS_MAP_MAIN.get(main_idx, "IFV")
            last_equip_label = equip_label

            # --- classificazione minaccia (Tank o IFV) ---
            threat_model = tank_model if equip_label == "Tank" else ifv_model
            x_threat = _preprocess_for_model(threat_model, frame)
            threat_raw = threat_model.predict(x_threat, verbose=0)[0]

            # percentuali (Low, Med, High) che sommano 100
            v = threat_raw.astype(np.float32).ravel()
            if np.any(v < 0) or not np.isclose(np.sum(v), 1.0, atol=1e-3):
                e = np.exp(v - np.max(v)); v = e / np.sum(e)
            v = (v / np.sum(v)) * 100.0
            perc = np.array([v[0], v[1], v[2]])  # [Low, Med, High]

            buf_low.append(perc[0]); buf_med.append(perc[1]); buf_high.append(perc[2])
            if len(buf_low) > smooth:
                buf_low.pop(0); buf_med.pop(0); buf_high.pop(0)

            smoothed = np.array([np.mean(buf_low), np.mean(buf_med), np.mean(buf_high)])
        else:
            equip_label = last_equip_label
            smoothed = np.array([np.mean(buf_low) if buf_low else 0,
                                 np.mean(buf_med) if buf_med else 0,
                                 np.mean(buf_high) if buf_high else 0])

        frame_ov = _draw_overlay_bottom_right(frame, equip_label, smoothed)
        out.write(frame_ov)
        i += 1

    cap.release()
    out.release()
