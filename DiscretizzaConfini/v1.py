import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def estrai_matrice_confine(image_path, dimensione=100, soglia=128):
    print(f"Elaborazione dell'immagine: {image_path}...")

    # 1. Caricamento in scala di grigi ("L")
    img = Image.open(image_path).convert("L")

    # 2. Ridimensionamento alla griglia desiderata (es. 100x100)
    img_scalata = img.resize((dimensione, dimensione))

    # 3. Conversione in array numerico (Matrice)
    arr = np.array(img_scalata)

    # 4. Discretizzazione (Binarizzazione)
    # Assumiamo che tu abbia disegnato un tratto scuro su un foglio chiaro.
    # Tutto ciò che è più scuro della 'soglia' diventa 1 (confine). Il resto diventa 0 (vuoto).
    matrice_2d = np.where(arr < soglia, 1, 0)

    return img_scalata, matrice_2d

# --- ESECUZIONE DEL TEST ---

# INSERISCI QUI IL NOME DELLA TUA FOTO
nome_file = "confini1.jpg"

if not os.path.exists(nome_file):
    print(f"ERRORE: Assicurati di avere un'immagine chiamata '{nome_file}' nella cartella!")
else:
    # Richiamiamo la funzione magica
    img_originale, matrice = estrai_matrice_confine(nome_file, dimensione=100)

    # Verifichiamo la forma della matrice (dovrebbe stampare: (100, 100))
    print(f"\nOperazione completata! Dimensione della matrice generata: {matrice.shape}")

    # --- VISUALIZZAZIONE DEI RISULTATI ---
    # Creiamo una finestra con due pannelli per vedere la trasformazione
    fig, assi = plt.subplots(1, 2, figsize=(10, 5))

    # Pannello Sinistro: La foto ridimensionata
    assi[0].imshow(img_originale, cmap="gray")
    assi[0].set_title(f"1. Foto Ridimensionata ({matrice.shape[0]}x{matrice.shape[1]})")
    assi[0].axis("off")

    # Pannello Destro: La matrice pura tradotta in grafica
    # Usiamo 'binary' per mappare lo 0 sul bianco e l'1 sul nero
    assi[1].imshow(matrice, cmap="binary")
    assi[1].set_title("2. Matrice Discretizzata (Solo 0 e 1)")
    assi[1].axis("off")

    plt.tight_layout()
    plt.show()
