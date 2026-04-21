
from pathlib import Path
import struct
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub
from tensorflow import keras
from tensorflow.keras import layers
 
# Per rendere i risultati più riproducibili
np.random.seed(42)
keras.utils.set_random_seed(42)
 
# Classi che ci interessano davvero per il progetto
TARGET_LABELS = ["S", "W", "A", "X", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
 
# Dizionari di conversione: label testuale <-> indice numerico
label_to_index = {label: idx for idx, label in enumerate(TARGET_LABELS)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
 
# Parametri principali, tenuti simili al notebook della prof
IMG_SIZE = 28
NUM_CLASSES = len(TARGET_LABELS)
FEATURE_VECTOR_LENGTH = IMG_SIZE * IMG_SIZE
 
# Download del dataset da Kaggle
path = kagglehub.dataset_download("crawford/emnist")
dataset_path = Path(path)
print("Path to dataset files:", dataset_path)
 
# --- Funzioni di caricamento IDX (formato binario EMNIST) ---
 
def load_idx_images(filepath):
    """Carica immagini in formato IDX (con o senza gzip)."""
    opener = gzip.open if str(filepath).endswith(".gz") else open
    with opener(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # EMNIST richiede trasposizione per orientamento corretto
    images = data.reshape(num, rows, cols).transpose(0, 2, 1).reshape(num, rows * cols)
    return images
 
def load_idx_labels(filepath):
    """Carica label in formato IDX (con o senza gzip)."""
    opener = gzip.open if str(filepath).endswith(".gz") else open
    with opener(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data
 
def load_mapping(filepath):
    """Carica il mapping emnist_idx -> carattere dal file di testo."""
    mapping = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx, ascii_code = int(parts[0]), int(parts[1])
                mapping[idx] = chr(ascii_code)
    return mapping
 
def find_file(base_path, pattern):
    """Cerca ricorsivamente un file per pattern glob."""
    matches = list(base_path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"File non trovato con pattern: {pattern}")
    return matches[0]
 
def filter_by_labels(X, y, mapping, target_labels):
    """Filtra solo i campioni delle classi target e rimappa le label a 0..N-1."""
    mask = np.array([mapping.get(int(lbl), "") in target_labels for lbl in y])
    X_filtered = X[mask]
    y_filtered = np.array([label_to_index[mapping[int(lbl)]] for lbl in y[mask]])
    return X_filtered, y_filtered
 
# --- Ricerca e caricamento file (split "balanced") ---
 
train_images_file = find_file(dataset_path, "*balanced*train*images*")
train_labels_file = find_file(dataset_path, "*balanced*train*labels*")
test_images_file  = find_file(dataset_path, "*balanced*test*images*")
test_labels_file  = find_file(dataset_path, "*balanced*test*labels*")
mapping_file      = find_file(dataset_path, "*balanced*mapping*")
 
print("Caricamento immagini e label...")
X_train_raw = load_idx_images(train_images_file)
y_train_raw = load_idx_labels(train_labels_file)
X_test_raw  = load_idx_images(test_images_file)
y_test_raw  = load_idx_labels(test_labels_file)
 
# Mapping EMNIST idx -> carattere (es. 1 -> '1', 10 -> 'A', ...)
emnist_mapping = load_mapping(mapping_file)
 
# Filtraggio: teniamo solo le classi in TARGET_LABELS
X_train, y_train = filter_by_labels(X_train_raw, y_train_raw, emnist_mapping, TARGET_LABELS)
X_test,  y_test  = filter_by_labels(X_test_raw,  y_test_raw,  emnist_mapping, TARGET_LABELS)
 
print("Shape train dopo reshape:", X_train.shape)
print("Shape test dopo reshape:", X_test.shape)
 
# Normalizzazione tra 0 e 1
X_train = X_train / 255.0
X_test  = X_test  / 255.0
 
# Modello MLP in stile prof
model = keras.Sequential()
model.add(layers.Input(shape=(FEATURE_VECTOR_LENGTH,)))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
model.summary()
 
# Parametri di training
batch_size = 128
epochs = 15
 
# Compile
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
 
# Training
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)
 
# Valutazione finale
score = model.evaluate(X_test, y_test, verbose=False)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
 
# Predizione su un esempio del test set
predict_x = model.predict(X_test[:1], verbose=False)
predicted_class_index = np.argmax(predict_x, axis=1)[0]
print("Probabilità:", predict_x[0])
print("Somma probabilità:", np.sum(predict_x[0]))
print("Classe predetta:", index_to_label[predicted_class_index])
print("Classe vera:", index_to_label[y_test[0]])
 
# Visualizza l'immagine di test e il risultato
x_test_vis = X_test[0].reshape(IMG_SIZE, IMG_SIZE)
plt.imshow(x_test_vis, cmap=plt.cm.binary)
plt.title(
    f"Vero: {index_to_label[y_test[0]]} - Predetto: {index_to_label[predicted_class_index]}"
)
plt.axis("off")
plt.show()
 
# Grafico accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper left")
plt.show()
 
# Grafico loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc="upper right")
plt.show()
 
# Salvataggio del modello addestrato
model.save("riconoscimentocaratteri.keras")
print("Modello salvato come: riconoscimentocaratteri.keras")