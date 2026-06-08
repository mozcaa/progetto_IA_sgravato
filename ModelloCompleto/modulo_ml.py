import numpy as np
from PIL import Image
from tensorflow import keras

print("[ML] Caricamento del modello in corso...")
model = keras.models.load_model("ModelloCNN.keras")
TARGET_LABELS = ["A", "C", "G", "F", "W", "X", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
print("[ML] Modello caricato con successo\n")

def riconosci_carattere(crop_binario):
    # Riceve in input crop_binario, un array che rappresenta il ritaglio di una lettera. Questo array
    # viene quindi convertito in un immagine
    img_pulita = Image.fromarray(crop_binario)

    # Ritagliamo gli spazi vuoti attorno alla lettera:
    # questa funzione cerca il rettangolo più piccolo possibile che racchiude tutti i pixel "attivi",
    # cioè quelli diversi dallo sfondo
    bbox = img_pulita.getbbox()
    if bbox:    # carattere trovato
        lettera_ritagliata = img_pulita.crop(bbox)  # Elimina i margini superflui attorno alla lettera
    else:   # se non è stato rilevato un carattere
        lettera_ritagliata = img_pulita     # Lasciamo l'immagine così com'è

    # Ridimensioniamo la lettera, impostando il lato lungo a 20 pixel e ridimensionando l'altro in
    # modo proporzionale.
    # 20 pixel è la dimensione default dei caratteri del dataset, collocati su una tela da 28 pixel
    w, h = lettera_ritagliata.size
    if w == 0 or h == 0:
        return "VUOTO"

    # Proporzione per i lati
    if w > h:
        nuovo_w, nuovo_h = 20, max(1, int(20 * h / w))
    else:
        nuovo_w, nuovo_h = max(1, int(20 * w / h)), 20

    lettera_scalata = lettera_ritagliata.resize((nuovo_w, nuovo_h), Image.Resampling.LANCZOS)
    # Il filtro LANCZOS preserva i dettagli geometrici del tratto. A differenza del nearest neighbor o
    # del Bilinear, questo filtro utilizza una formula basata sul sinc, e analizza una matrice 6x6 di
    # pixel vicini per calcolare il valore del nuovo pixel. Utilizziamo questo filtro perché preserva
    # l'integrità geometrica dei caratteri. Riduce l'aliasing e mantiene la nitidezza dei bordi.

    # Incolliamo al centro della tela 28x28
    tela_finale = Image.new('L', (28, 28), color=0) # Creazione di una tela nera ("L" sta per canale singolo (scala di grigi))
    x_offset = (28 - nuovo_w) // 2
    y_offset = (28 - nuovo_h) // 2
    tela_finale.paste(lettera_scalata, (x_offset, y_offset))

    # Conversione del file in float32 e normalizzazione dei colori (grigi) tra 0.0 e 1.0
    # L'intervallo ristretto rende i calcoli matematici più della rete più stabili e veloci
    img_array = np.array(tela_finale, dtype=np.float32) / 255.0
    # Matrice (immagine) modellata in (1, 28, 28, 1)
    # Questo indica un batch contenente una sola immagine, alta 28, larga 28 e 1 solo canale colore (scala di grigi)
    img_array_cnn = img_array.reshape(1, 28, 28, 1)

    # Passiamo l'immagine modificata al modello. pred è una matrice (1x15) di probabilità, contente numeri decimali.
    # Con verbose falso disattiviamo la barra di avanzamento (ingombra solo il terminale sennò)
    pred = model.predict(img_array_cnn, verbose=False)
    # Dal vettore di probabilità, argmax restituisce l'indice della probabilità più alta. Con axis=1,
    # indichiamo a python di muoversi in orizzontale lungo le colonne di quella riga.
    # Lo [0] in fondo serve a estrarre il primo elemento dell'array fornito da argmax
    predicted_index = np.argmax(pred, axis=1)[0]

    # Ritorniamo il carattere predetto, utilizzando l'indice trovato nella riga precedente
    return TARGET_LABELS[predicted_index]
