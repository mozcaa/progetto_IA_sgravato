import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Parametri modificabili
medianBlur = 11         # Se non vengono rilevati dei confini, aumentare questo valore (solo dispari)
adaptiveThreshold = 10  # Più è basso, più è sensibile ai tratti chiari
iterazioni = 1
dimensions = 200

def estrai_matrice_robusta(image_path, dimensione=dimensions):
    print(f"Elaborazione dell'immagine: {image_path}...")

    # 1. Caricamento in scala di grigi con OpenCV
    # Sostituisce PIL. Legge direttamente l'immagine in bianco e nero.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Immagine non trovata! Controlla il nome del file.")

    # --- NUOVO: ISPESSIMENTO DELLA PENNA ---
    # Creiamo un microscopico pennello 2x2
    kernel_penna = np.ones((2, 2), np.uint8)
    # L'erosione in scala di grigi rende i tratti scuri più spessi!
    img = cv2.erode(img, kernel_penna, iterations=1)
    # ---------------------------------------

    # 2. Rimozione Quadretti (Filtro Mediano)
    # Sostituisce ogni pixel con la 'mediana' dei pixel vicini.
    # Cancella la griglia sottile ma salva il pennarello spesso. (11 è la forza del filtro)
    img_sfocata = cv2.medianBlur(img, medianBlur)

    # 3. Rimozione Ombre (Soglia Adattiva)
    # Calcola il contrasto a zone. THRESH_BINARY_INV fa sì che il tratto diventi 255 (bianco) e lo sfondo 0.
    img_binaria = cv2.adaptiveThreshold(
        img_sfocata,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, # Dimensione della zona di calcolo (deve essere dispari. 31 tollera ombre grandi)
        adaptiveThreshold  # Costante di tolleranza (se vedi ancora "sporco" alza questo numero a 15 o 20)
    )

    # --- NUOVO: DILATAZIONE (Morfologia) ---
    # Creiamo un "pennello" di 3x3 pixel
    kernel = np.ones((3,3), np.uint8)
    # Spalmiamo l'inchiostro rilevato per unire i buchi della penna
    img_binaria = cv2.dilate(img_binaria, kernel, iterations=iterazioni)
    # ---------------------------------------

    # 4. Pulizia finale (Tieni solo l'isola più grande)
    # Trova tutti i gruppi di pixel e tiene solo il tuo confine, cancellando la polvere.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)
    if num_labels > 1:
        # Prende l'indice dell'isola più grande (escludendo lo sfondo che è 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        img_pulita = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        img_pulita = img_binaria

    # 5. Ridimensionamento alla griglia finale (100x100)
    img_scalata = cv2.resize(img_pulita, (dimensione, dimensione), interpolation=cv2.INTER_AREA)

    # 6. Discretizzazione in Matrice 0 e 1
    # Tutto ciò che è grigio scuro/bianco diventa 1 (confine).
    matrice_2d = np.where(img_scalata > 127, 1, 0)

    return img, matrice_2d

# --- ESECUZIONE ---

nome_file = "confini.jpg" # INSERISCI QUI LA TUA FOTO

if not os.path.exists(nome_file):
    print(f"ERRORE: Assicurati di avere un'immagine chiamata '{nome_file}'.")
else:
    img_originale, matrice = estrai_matrice_robusta(nome_file, dimensione=dimensions)

    print(f"\nOperazione completata! Dimensione matrice: {matrice.shape}")

    # Visualizzazione
    fig, assi = plt.subplots(1, 2, figsize=(10, 5))

    assi[0].imshow(img_originale, cmap="gray")
    assi[0].set_title("1. Foto Originale (Senza Flash / Quadretti)")
    assi[0].axis("off")

    assi[1].imshow(matrice, cmap="binary")
    assi[1].set_title("2. Matrice Discretizzata e Pulita")
    assi[1].axis("off")

    plt.tight_layout()
    plt.show()
