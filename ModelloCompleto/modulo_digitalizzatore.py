import cv2
import numpy as np

# I tuoi parametri fedeli al codice originale
MEDIAN_BLUR = 15
ADAPTIVE_THRESH = 10
ITERAZIONI_DILATAZIONE = 1

def ottieni_matrice_binaria(image_path):
    print(f"[Digitalizzatore] Elaborazione dell'immagine: {image_path}...")

    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Immagine {image_path} non trovata.")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Ispessimento della penna
    kernel_penna = np.ones((2, 2), np.uint8)
    img_gray = cv2.erode(img_gray, kernel_penna, iterations=1)

    # Rimozione quadretti
    img_sfocata = cv2.medianBlur(img_gray, MEDIAN_BLUR)

    # Rimozione ombre
    img_binaria = cv2.adaptiveThreshold(
        img_sfocata, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, ADAPTIVE_THRESH
    )

    # Dilatazione (Morfologia) per unire i buchi
    kernel = np.ones((3,3), np.uint8)
    img_binaria = cv2.dilate(img_binaria, kernel, iterations=ITERAZIONI_DILATAZIONE)

    return img_color, img_binaria
