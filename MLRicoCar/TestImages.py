import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from PIL import Image, ImageOps
from tensorflow import keras

# 1. Configurazione iniziale
TARGET_LABELS = ["A", "C", "G", "F", "W", "X", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
index_to_label = {idx: label for idx, label in enumerate(TARGET_LABELS)}
IMG_SIZE = 28
FOLDER_PATH = "ImmaginiTest"

# --- LA MAGIA: Funzione per replicare il formato EMNIST ---
def centra_e_ridimensiona(img_path):
    img = Image.open(img_path).convert("L")
    img = ImageOps.invert(img)
    
    arr = np.array(img)
    
    # --- LA MAGIA SCIPY: TROVA E TIENI SOLO L'ISOLA PIÙ GRANDE ---
    # 1. Creiamo una maschera dei pixel accesi
    mask = arr > 20 
    
    # 2. Etichettiamo tutti i gruppi di pixel separati (le "isole")
    labels, num_features = ndi.label(mask)
    
    if num_features > 0:
        # Calcoliamo quanti pixel ha ogni isola
        sizes = ndi.sum(mask, labels, range(1, num_features + 1))
        
        # Troviamo l'indice dell'isola più grande (la nostra lettera)
        largest_label = np.argmax(sizes) + 1
        
        # AZZERIAMO TUTTO ciò che non appartiene all'isola più grande! Addio puntini.
        arr = np.where(labels == largest_label, arr, 0)
    # -------------------------------------------------------------
    
    # Ora calcoliamo il ritaglio sull'array perfettamente pulito
    righe_con_pixel = np.any(arr > 0, axis=1)
    colonne_con_pixel = np.any(arr > 0, axis=0)
    
    if not np.any(righe_con_pixel) or not np.any(colonne_con_pixel):
        return img.resize((28, 28))
        
    y_min, y_max = np.where(righe_con_pixel)[0][[0, -1]]
    x_min, x_max = np.where(colonne_con_pixel)[0][[0, -1]]
    
    # Creiamo una nuova immagine PIL usando l'array pulito senza rumore
    img_pulita = Image.fromarray(arr.astype(np.uint8))
    
    # Ritagliamo
    lettera_ritagliata = img_pulita.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Ridimensioniamo (Lato lungo = 20)
    w, h = lettera_ritagliata.size
    if w > h:
        nuovo_w, nuovo_h = 20, max(1, int(20 * h / w))
    else:
        nuovo_w, nuovo_h = max(1, int(20 * w / h)), 20
        
    lettera_scalata = lettera_ritagliata.resize((nuovo_w, nuovo_h), Image.Resampling.LANCZOS)
    
    # Incolliamo al centro
    tela_finale = Image.new('L', (28, 28), color=0)
    x_offset = (28 - nuovo_w) // 2
    y_offset = (28 - nuovo_h) // 2
    tela_finale.paste(lettera_scalata, (x_offset, y_offset))
    
    return tela_finale
# -----------------------------------------------------------

# 2. Caricamento del modello (assicurati che il nome sia quello corretto!)
print("Caricamento del modello in corso...")
model = keras.models.load_model("ModelloCNN.keras")
print("Modello caricato con successo!\n")

if not os.path.exists(FOLDER_PATH):
    print(f"Errore: La cartella '{FOLDER_PATH}' non esiste in questo percorso.")
else:
    # 3. Ciclo su tutti i file
    for filename in os.listdir(FOLDER_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(FOLDER_PATH, filename)
            
            print(f"Analizzando: {filename}...")
            
            # Applichiamo la nostra nuova funzione di pre-processing
            img_processata = centra_e_ridimensiona(image_path)
            
            img_array = np.array(img_processata, dtype=np.float32)
            img_array = img_array / 255.0
            
            # Reshape per la CNN
            img_array_cnn = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            
            # Predizione
            pred = model.predict(img_array_cnn, verbose=False)
            predicted_index = np.argmax(pred, axis=1)[0]
            predicted_label = index_to_label[predicted_index]
            
            # Visualizzazione
            plt.figure(figsize=(4, 4))
            plt.imshow(img_array, cmap="gray")
            plt.title(f"File: {filename}\nPredetto: {predicted_label}", fontsize=14)
            plt.axis("off")
            plt.show()

print("Test su tutte le immagini completato!")