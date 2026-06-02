# L'immagine di input sarà a colori, mentre quella di output sarà una matrice di valori da 0 a 255
# in scala di grigi (0 = nero e 255 = bianco)

import cv2  # Essenziale per processare le immagini
import numpy as np  # Usata per gestire le matrici numeriche, essenziali per operazioni su immagini

# Qui andiamo a definire i parametri per la digitalizzazione dell'immagine
MEDIAN_BLUR = 3             # Blur applicato alle righe. Serve per aiutare il modello di ML e per chiudere
                            # eventuali buchi minuscoli dati dal rumore.
ADAPTIVE_THRESH = 10        # Regola il Gaussian_thresholding, indica la sensibilità tramite la quale
                            # diversifichiamo un insieme di pixel neri da ombre a righe effettive.
ITERAZIONI_DILATAZIONE = 1  # Il numero di iterazioni che il codice deve eseguire per la dilatazione delle
                            # righe visibili, in modo da chiudere il maggior numero di buchi possibili

def ottieni_matrice_binaria(image_path):
    print(f"[Digitalizzatore] Elaborazione dell'immagine: {image_path}...")

    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Immagine {image_path} non trovata.")

    # Convertiamo l'immagine in scala di grigi (la penna è scura)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Ispessimento della penna (assegna il valore minimo del kernel)
    kernel_penna = np.ones((3, 3), np.uint8)    # Creazione matrice unitaria 3x3.
    img_gray = cv2.erode(img_gray, kernel_penna, iterations=1)

    # Applichiamo un blur per rimuovere il rumore (come i quadretti del foglio).
    # Calcola la media dei pixel in un'area
    img_sfocata = cv2.medianBlur(img_gray, MEDIAN_BLUR)

    # Trasforma l'immagine blurrata in binaria: utilizza la tecnica Adaptive Thresholding per stabilire
    # una soglia di luminosità che varia a seconda dell'area circostante.
    img_binaria = cv2.adaptiveThreshold(
        img_sfocata, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, ADAPTIVE_THRESH
    )
    # Qui invertiamo anche i colori dell'immagine, avendo la penna come bianco e lo sfondo come nero, 
    # impostando tutti i valori che sono sotto una certa soglia (ovvero nero) al valore massimo 255, e viceversa,
    # perché la dilatazione (funzione successiva) lavora espandendo le zone con valori massimi (ovvero bianco).

    # Dilatazione (morfologia) per unire i buchi: tramite una matrice 3x3 ingrandisce gli oggetti
    # binari (in questo caso le tracce della penna), creando un'immagine più definita.
    kernel = np.ones((3,3), np.uint8)
    img_binaria = cv2.dilate(img_binaria, kernel, iterations=ITERAZIONI_DILATAZIONE)

    return img_color, img_binaria
