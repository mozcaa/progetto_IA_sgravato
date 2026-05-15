import cv2
import numpy as np
import matplotlib.pyplot as plt

import modulo_digitalizzatore
import modulo_ml
import pathfinding

DIMENSIONE_FINALE = 300

AREA_RUMORE = 400
AREA_CONFINE = 15000

COLORI_PER_LABEL = {
    "1": [255, 100, 100], # Rosso
    "2": [100, 255, 100], # Verde
    "3": [100, 100, 255], # Blu
    "4": [255, 255, 100], # Giallo
    "5": [255, 100, 255], # Magenta
    "6": [100, 255, 255], # Ciano
    "7": [255, 165, 0],   # Arancione
    "8": [147, 112, 219], # Viola
    "9": [255, 192, 203], # Rosa
    "C": [0, 250, 154],   # Verde primavera
    "W": [22, 230, 140], # Khaki
    "X": [173, 216, 230], # Azzurro
    "A": [223, 0, 100], # Qualcosa
    "G": [142, 93, 240], # Qualcosa 2
    "F": [93, 142, 240], # Qualcosa 3
    "VUOTO": [0, 0, 0]
}


def avvia_hub(image_path):
    print(f"\n--- AVVIO HUB CENTRALE ---")

    img_color, img_binaria = modulo_digitalizzatore.ottieni_matrice_binaria(image_path)

    step1_visiva = cv2.resize(img_binaria, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("step1_matrice_completa.png", cv2.bitwise_not(step1_visiva))
    print("- Salvato 'step1_matrice_completa.png'")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)

    maschera_confini = np.zeros_like(img_binaria)
    caratteri_riconosciuti = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > AREA_CONFINE:
            maschera_confini[labels == i] = 255

        elif area > AREA_RUMORE:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            crop = np.where(labels[y:y+h, x:x+w] == i, 255, 0).astype(np.uint8)
            centro = (int(centroids[i][0]), int(centroids[i][1]))

            label_predetta = modulo_ml.riconosci_carattere(crop)
            caratteri_riconosciuti.append({'centro': centro, 'label': label_predetta})

            cv2.putText(img_color, label_predetta, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

    step2_visiva = cv2.resize(maschera_confini, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("step2_solo_confini.png", cv2.bitwise_not(step2_visiva))
    print("- Salvato 'step2_solo_confini.png'")

    # --- NUOVA LOGICA DI DIVISIONE STATI ---
    # 1. Invertiamo i confini: i muri diventano neri (0), e gli spazi vuoti bianchi (255)
    aree_senza_confini = cv2.bitwise_not(maschera_confini)

    # 2. Troviamo tutte le "pozze" separate (gli stati)
    numero_aree, labels_aree, stats_aree, centroidi_aree = cv2.connectedComponentsWithStats(aree_senza_confini, connectivity=4)

    matrice_id_territori = np.zeros_like(img_binaria, dtype=np.int32)
    label_per_id_territorio = {}
    id_territorio = 1

    # 3. L'oceano (lo spazio esterno fuori dalla mappa) tocca sicuramente l'angolo (0,0)
    label_esterno = labels_aree[0, 0]

    for i in range(1, numero_aree):
        if i == label_esterno:
            continue # Saltiamo l'oceano

        label_territorio = "VUOTO"
        for char in caratteri_riconosciuti:
            cx, cy = char['centro']
            # Se il centro della lettera cade esattamente in questa "pozza" (stato i)
            if labels_aree[cy, cx] == i:
                label_territorio = char['label']
                break

        # Scriviamo i dati in memoria
        matrice_id_territori[labels_aree == i] = id_territorio
        label_per_id_territorio[id_territorio] = label_territorio
        id_territorio += 1

    # Ridimensionamento
    confini_ridimensionati = cv2.resize(maschera_confini, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)
    territori_ridimensionati = cv2.resize(matrice_id_territori, (DIMENSIONE_FINALE, DIMENSIONE_FINALE), interpolation=cv2.INTER_NEAREST)

    # Creazione della Matrice 3D Finale
    matrice_3d = np.empty((DIMENSIONE_FINALE, DIMENSIONE_FINALE, 2), dtype=object)
    immagine_matrice_colorata = np.zeros((DIMENSIONE_FINALE, DIMENSIONE_FINALE, 3), dtype=np.uint8)

    for y in range(DIMENSIONE_FINALE):
        for x in range(DIMENSIONE_FINALE):
            if confini_ridimensionati[y, x] > 127:
                matrice_3d[y, x, 0] = 2
                matrice_3d[y, x, 1] = "CONFINE"
                immagine_matrice_colorata[y, x] = [0, 0, 0]
            else:
                id_territorio_corrente = territori_ridimensionati[y, x]
                if id_territorio_corrente > 0:
                    matrice_3d[y, x, 0] = 1
                    matrice_3d[y, x, 1] = label_per_id_territorio[id_territorio_corrente]

                    # Seleziona il colore associato alla label riconosciuta
                    colore_scelto = COLORI_PER_LABEL[label_per_id_territorio[id_territorio_corrente]]
                    immagine_matrice_colorata[y, x] = colore_scelto
                else:
                    matrice_3d[y, x, 0] = 0
                    matrice_3d[y, x, 1] = "VUOTO"
                    immagine_matrice_colorata[y, x] = [255, 255, 255]

    # --- SCALATURA E RICERCA START/GOAL ---
    
    # 1. Calcoliamo di quanto abbiamo ingrandito la mappa, quindi le scale che ci servono
    altezza_originale, larghezza_originale = img_binaria.shape
    scala_x = DIMENSIONE_FINALE / larghezza_originale
    scala_y = DIMENSIONE_FINALE / altezza_originale

    start_scalato = None
    goal_scalato = None

    # 2. Scorriamo i caratteri trovati, cerchiamo C e W
    for char in caratteri_riconosciuti:
        cx_orig, cy_orig = char['centro']
        
        if char['label'] == 'C':
            start_scalato = (int(cx_orig * scala_x), int(cy_orig * scala_y)) #trovata la C salvo le coordinate scalate del suo baricentro
        elif char['label'] == 'W':
            goal_scalato = (int(cx_orig * scala_x), int(cy_orig * scala_y)) #trovata la W salvo le coordinate scalate del suo baricentro

    if not start_scalato or not goal_scalato:
        print("ATTENZIONE: Commence (C) o Win (W) non trovati!")
    
    if start_scalato: #metto un punto verde sulla mappa colorata sulla posizione di C
        # cv2.circle(immagine, (x, y), raggio, colore_rgb, spessore)
        # Lo spessore -1 riempie completamente il cerchio
        cv2.circle(immagine_matrice_colorata, start_scalato, 3, (0, 255, 0), -1) 

    if goal_scalato: #metto un punto rosso sulla mappa colorata sulla posizione di W
        cv2.circle(immagine_matrice_colorata, goal_scalato, 3, (255, 0, 0), -1)

    cv2.imwrite("step3_matrice_colorata.png", cv2.cvtColor(immagine_matrice_colorata, cv2.COLOR_RGB2BGR))
    print("- Salvato 'step3_matrice_colorata.png'")
    print("--- ELABORAZIONE COMPLETATA ---")

    return img_color, immagine_matrice_colorata, matrice_3d, start_scalato, goal_scalato


if __name__ == "__main__":
    nome_foto = "mappe/mappa16.jpg"  # Controlla sempre che il nome combaci!

    img_originale, immagine_matrice_colorata, mat_3d, start, goal = avvia_hub(nome_foto)
    
    # ATTIVAZIONE PATHFINDING E RAPPRESENTAZIONE PERCORSO SU MAPPA

    if start and goal:
     
        print("Colori percorsi:")
        print("A* Manhattan = rosso")
        print("A* Euclideo = giallo")
        print("UCS = verde")
        print("Greedy Manhattan = blu")
        print("Greedy Euclideo = viola/magenta")
        # Passiamo la matrice e i punti scalati al file separato
        percorso_ucs, percorso_astar_manhattan, percorso_astar_euclideo, percorso_greedy_manhattan, percorso_greedy_euclideo = pathfinding.esegui_confronto(mat_3d, start, goal)
        
        # DISEGNARE IL PERCORSO SULLA MAPPA
        # Se A* Manhattan ha trovato un percorso, coloriamo i pixel del percorso di rosso sulla mappa finale
        if percorso_astar_manhattan:
            for x, y in percorso_astar_manhattan:
                # Coloriamo il pixel di rosso [Rosso, Verde, Blu]
                immagine_matrice_colorata[y, x] = [255, 0, 0]
                immagine_matrice_colorata[y+1, x+1] = [255, 0, 0]

        if percorso_astar_euclideo:
            for x, y in percorso_astar_euclideo:
                # Coloriamo il pixel di giallo [Rosso, Verde, Blu]
                immagine_matrice_colorata[y, x] = [255, 255, 0]
        # Se UCS ha trovato un percorso, coloriamo i pixel del percorso di verde sulla mappa finale

        if percorso_ucs:
            for x, y in percorso_ucs:
                # Coloriamo il pixel di verde [Rosso, Verde, Blu]
                immagine_matrice_colorata[y, x] = [0, 255, 0]
        # Se Greedy ha trovato un percorso, coloriamo i pixel del percorso di blu sulla mappa finale
        
        if percorso_greedy_manhattan:
            for x, y in percorso_greedy_manhattan:
                # Coloriamo il pixel di blu [Rosso, Verde, Blu]
                immagine_matrice_colorata[y, x] = [0, 0, 255]

        if percorso_greedy_euclideo:
            for x, y in percorso_greedy_euclideo:
                # Coloriamo il pixel di viola [Rosso, Verde, Blu]
                immagine_matrice_colorata[y, x] = [255, 0, 255]
                
    fig, assi = plt.subplots(1, 2, figsize=(12, 6))
    assi[0].imshow(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB))
    assi[0].set_title("1. Mappa Analizzata", fontsize=14)
    assi[0].axis("off")

    assi[1].imshow(immagine_matrice_colorata)
    assi[1].set_title(f"2. Stati Indipendenti ({DIMENSIONE_FINALE}x{DIMENSIONE_FINALE})", fontsize=14)
    assi[1].axis("off")

    plt.tight_layout()
    plt.show()