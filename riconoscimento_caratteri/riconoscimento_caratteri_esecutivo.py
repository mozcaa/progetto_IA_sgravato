import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras

# Le tue classi
TARGET_LABELS = ["S", "W", "A", "X", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
index_to_label = {idx: label for idx, label in enumerate(TARGET_LABELS)}

IMG_SIZE = 28

# Carica il modello salvato
model = keras.models.load_model("riconoscimentocaratteri.keras")

# Percorso della tua immagine
image_path = "6.png"   

# Carica immagine e convertila in scala di grigi
img = Image.open(image_path).convert("L")

# Ridimensiona a 28x28
img = img.resize((IMG_SIZE, IMG_SIZE))

# Se necessario, inverti i colori
img = ImageOps.invert(img)

# Converti in array numpy
img_array = np.array(img, dtype=np.float32)

# Normalizza tra 0 e 1
img_array = img_array / 255.0

# Appiattisci: da 28x28 a vettore di 784 elementi
img_array_flat = img_array.reshape(1, IMG_SIZE * IMG_SIZE)

# Predizione
pred = model.predict(img_array_flat, verbose=False)
predicted_index = np.argmax(pred, axis=1)[0]
predicted_label = index_to_label[predicted_index]

print("Probabilità:", pred[0])
print("Classe predetta:", predicted_label)

# Visualizza l'immagine usata dal modello
plt.imshow(img_array, cmap="gray")
plt.title(f"Predetto: {predicted_label}")
plt.axis("off")
plt.show()