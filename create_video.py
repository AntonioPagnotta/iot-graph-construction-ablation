import cv2
import os
import glob
from tqdm.auto import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================
FRAMES_DIR = os.path.join('video_frames', 'frames')
OUTPUT_VIDEO = os.path.join('video_frames', 'network_animation.mp4')
FPS = 3  # Fotogrammi al secondo (modifica questo valore per velocizzare o rallentare il video)


def create_video_from_frames():
    print(f"Ricerca frame in: {FRAMES_DIR}")

    # Cerca tutti i file PNG nella cartella
    search_pattern = os.path.join(FRAMES_DIR, "*.png")
    frames = glob.glob(search_pattern)

    # Ordina i file in base al nome.
    # Essendo i nomi nel formato "YYYY-MM-DD_HH-MM-SS.png",
    # l'ordine alfabetico coinciderà esattamente con l'ordine cronologico.
    frames.sort()

    if not frames:
        print("ERRORE: Nessun frame (.png) trovato nella cartella specificata.")
        return

    print(f"Trovati {len(frames)} frames. Creazione del video...")

    # Leggi il primo frame per ottenere le dimensioni del video (larghezza, altezza)
    first_frame = cv2.imread(frames[0])
    height, width, layers = first_frame.shape
    size = (width, height)

    # Inizializza il VideoWriter di OpenCV
    # Usiamo il codec 'mp4v' che genera file .mp4 compatibili
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, size)

    # Aggiungi ogni frame al video
    for frame_path in tqdm(frames, desc="Compilazione Video"):
        img = cv2.imread(frame_path)
        out.write(img)

    # Rilascia le risorse e chiudi il file video
    out.release()
    print(f"\nVideo completato con successo! Salvato in: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    create_video_from_frames()