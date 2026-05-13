import numpy as np
from PIL import Image
from tensorflow import keras

print("[ML] Caricamento del modello in corso...")
model = keras.models.load_model("ModelloCNN.keras")
TARGET_LABELS = ["C", "W", "A", "X", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
print("[ML] Modello caricato con successo!\n")

def riconosci_carattere(crop_binario):
    """Prende un array numpy (la lettera ritagliata dall'Hub) e la riconosce."""
    img_pulita = Image.fromarray(crop_binario)

    # Ritagliamo gli spazi vuoti attorno
    bbox = img_pulita.getbbox()
    if bbox:
        lettera_ritagliata = img_pulita.crop(bbox)
    else:
        lettera_ritagliata = img_pulita

    # Ridimensioniamo (Lato lungo = 20) come nel tuo codice originale
    w, h = lettera_ritagliata.size
    if w == 0 or h == 0:
        return "VUOTO"

    if w > h:
        nuovo_w, nuovo_h = 20, max(1, int(20 * h / w))
    else:
        nuovo_w, nuovo_h = max(1, int(20 * w / h)), 20

    lettera_scalata = lettera_ritagliata.resize((nuovo_w, nuovo_h), Image.Resampling.LANCZOS)

    # Incolliamo al centro della tela 28x28
    tela_finale = Image.new('L', (28, 28), color=0)
    x_offset = (28 - nuovo_w) // 2
    y_offset = (28 - nuovo_h) // 2
    tela_finale.paste(lettera_scalata, (x_offset, y_offset))

    # Reshape e normalizzazione per la CNN
    img_array = np.array(tela_finale, dtype=np.float32) / 255.0
    img_array_cnn = img_array.reshape(1, 28, 28, 1)

    # Predizione
    pred = model.predict(img_array_cnn, verbose=False)
    predicted_index = np.argmax(pred, axis=1)[0]

    return TARGET_LABELS[predicted_index]
