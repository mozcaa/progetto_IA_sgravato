import cv2
import numpy as np
import matplotlib.pyplot as plt

import modulo_digitalizzatore
import modulo_ml

DIMENSIONE_FINALE = 1280

AREA_RUMORE = 400
AREA_CONFINE = 15000

PALETTE = [
    [255, 100, 100], # Rosso
    [100, 255, 100], # Verde
    [100, 100, 255], # Blu
    [255, 255, 100], # Giallo
    [255, 100, 255], # Magenta
    [100, 255, 255], # Ciano
    [255, 165, 0],   # Arancione
    [147, 112, 219], # Viola
    [255, 192, 203], # Rosa
    [0, 250, 154],   # Verde primavera
    [240, 230, 140], # Khaki
    [173, 216, 230]  # Azzurro
]

def avvia_hub(image_path):
    print(f"\n--- AVVIO HUB CENTRALE ---")

    img_color, img_binaria = modulo_digitalizzatore.ottieni_matrice_binaria(image_path)

    step1_visiva = cv2.resize(img_binaria, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("step1_matrice_completa.png", cv2.bitwise_not(step1_visiva))
    print("- Salvato 'step1_matrice_completa.png'")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)

    mask_confini = np.zeros_like(img_binaria)
    lista_caratteri = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > AREA_CONFINE:
            mask_confini[labels == i] = 255

        elif area > AREA_RUMORE:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            crop = np.where(labels[y:y+h, x:x+w] == i, 255, 0).astype(np.uint8)
            centro = (int(centroids[i][0]), int(centroids[i][1]))

            label_predetta = modulo_ml.riconosci_carattere(crop)
            lista_caratteri.append({'centro': centro, 'label': label_predetta})

            cv2.putText(img_color, label_predetta, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

    step2_visiva = cv2.resize(mask_confini, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("step2_solo_confini.png", cv2.bitwise_not(step2_visiva))
    print("- Salvato 'step2_solo_confini.png'")

    # --- NUOVA LOGICA DI DIVISIONE STATI ---
    # 1. Invertiamo i confini: i muri diventano neri (0), e gli spazi vuoti bianchi (255)
    sfondo = cv2.bitwise_not(mask_confini)

    # 2. Troviamo tutte le "pozze" separate (gli stati)
    num_stati, labels_stati, stats_stati, centroids_stati = cv2.connectedComponentsWithStats(sfondo, connectivity=4)

    mappa_territori = np.zeros_like(img_binaria, dtype=np.int32)
    dizionario_colori = {}
    id_colore = 1

    # 3. L'oceano (lo spazio esterno fuori dalla mappa) tocca sicuramente l'angolo (0,0)
    label_esterno = labels_stati[0, 0]

    for i in range(1, num_stati):
        if i == label_esterno:
            continue # Saltiamo l'oceano

        label_territorio = "VUOTO"
        for char in lista_caratteri:
            cx, cy = char['centro']
            # Se il centro della lettera cade esattamente in questa "pozza" (stato i)
            if labels_stati[cy, cx] == i:
                label_territorio = char['label']
                break

        # Scriviamo i dati in memoria
        mappa_territori[labels_stati == i] = id_colore
        dizionario_colori[id_colore] = label_territorio
        id_colore += 1

    # Ridimensionamento
    confini_ridim = cv2.resize(mask_confini, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    territori_ridim = cv2.resize(mappa_territori, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)

    # Creazione della Matrice 3D Finale
    matrice_3d = np.empty((DIMENSIONE_FINALE, DIMENSIONE_FINALE, 2), dtype=object)
    visualizzazione = np.zeros((DIMENSIONE_FINALE, DIMENSIONE_FINALE, 3), dtype=np.uint8)

    for y in range(DIMENSIONE_FINALE):
        for x in range(DIMENSIONE_FINALE):
            if confini_ridim[y, x] > 127:
                matrice_3d[y, x, 0] = 2
                matrice_3d[y, x, 1] = "CONFINE"
                visualizzazione[y, x] = [0, 0, 0]
            else:
                id_ter = territori_ridim[y, x]
                if id_ter > 0:
                    matrice_3d[y, x, 0] = 1
                    matrice_3d[y, x, 1] = dizionario_colori[id_ter]

                    # Seleziona un colore unico attingendo dalla palette
                    colore_scelto = PALETTE[(id_ter - 1) % len(PALETTE)]
                    visualizzazione[y, x] = colore_scelto
                else:
                    matrice_3d[y, x, 0] = 0
                    matrice_3d[y, x, 1] = "VUOTO"
                    visualizzazione[y, x] = [255, 255, 255]

    cv2.imwrite("step3_matrice_colorata.png", cv2.cvtColor(visualizzazione, cv2.COLOR_RGB2BGR))
    print("- Salvato 'step3_matrice_colorata.png'")
    print("--- ELABORAZIONE COMPLETATA ---")

    return img_color, visualizzazione, matrice_3d

if __name__ == "__main__":
    nome_foto = "mappa.jpg"  # Controlla sempre che il nome combaci!

    img_originale, img_matrice, mat_3d = avvia_hub(nome_foto)

    fig, assi = plt.subplots(1, 2, figsize=(12, 6))
    assi[0].imshow(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB))
    assi[0].set_title("1. Mappa Analizzata", fontsize=14)
    assi[0].axis("off")

    assi[1].imshow(img_matrice)
    assi[1].set_title(f"2. Stati Indipendenti ({DIMENSIONE_FINALE}x{DIMENSIONE_FINALE})", fontsize=14)
    assi[1].axis("off")

    plt.tight_layout()
    plt.show()
