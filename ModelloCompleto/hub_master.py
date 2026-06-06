import cv2
import numpy as np
import matplotlib.pyplot as plt

import modulo_digitalizzatore
import modulo_ml
import pathfinding

DIMENSIONE_FINALE = 300

AREA_RUMORE = 400
AREA_CONFINE = 15000

COLORI_PER_LABEL = { #sfumature di rosso
    "1": [255, 235, 235], # Rosa chiarissimo
    "2": [255, 205, 205],
    "3": [255, 170, 170],
    "4": [255, 130, 130],
    "5": [255, 85, 85],   # Rosso medio
    "6": [230, 40, 40],
    "7": [190, 0, 0],
    "8": [140, 0, 0],
    "9": [90, 0, 0],  # Bordeaux scurissimo
    "C": [0, 200, 0],       # Verde Scuro
    "W": [255, 215, 0],     # Oro/Giallo
    "X": [100, 100, 100],   # Grigio scuro
    "A": [200, 255, 200],   # Verde menta chiarissimo
    "G": [178, 255, 255],   # Celeste ghiaccio
    "F": [138, 43, 226],    # Viola acceso (BlueViolet)
    "VUOTO": [0, 0, 0]
}

# COLORI_PER_LABEL = { #heatmap
#     "1": [255, 255, 150], # Giallo chiaro
#     "2": [255, 235, 100], # Giallo intenso
#     "3": [255, 210, 50],  # Giallo-Arancio
#     "4": [255, 170, 0],   # Arancione
#     "5": [255, 120, 0],   # Arancione scuro
#     "6": [255, 70, 0],    # Rosso-Arancio
#     "7": [220, 20, 0],    # Rosso
#     "8": [170, 0, 0],     # Rosso scuro
#     "9": [120, 0, 0],
#     "C": [0, 255, 128],     # Verde Primavera
#     "W": [30, 144, 255],    # Blu Dodger (Azzurro intenso)
#     "X": [80, 80, 80],      # Antracite (Grigio quasi nero)
#     "A": [176, 224, 230],   # Powder Blue (Azzurrino polvere)
#     "G": [240, 255, 255],   # Bianco Ghiaccio
#     "F": [255, 0, 255],     # Magenta puro:
# }


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

    # ***** NUOVA LOGICA DI DIVISIONE STATI *****
    #  Invertiamo i confini: i muri diventano neri (0), e gli spazi vuoti bianchi (255)
    aree_senza_confini = cv2.bitwise_not(maschera_confini)

    # Troviamo tutte le "pozze" separate (gli stati)
    numero_aree, labels_aree, stats_aree, centroidi_aree = cv2.connectedComponentsWithStats(aree_senza_confini, connectivity=4)

    matrice_id_territori = np.zeros_like(img_binaria, dtype=np.int32)
    label_per_id_territorio = {}
    id_territorio = 1

    # L'oceano (lo spazio esterno fuori dalla mappa) tocca sicuramente l'angolo (0,0)
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
                matrice_3d[y, x, 1] = "CONFINE"
                immagine_matrice_colorata[y, x] = [0, 0, 0]
            else:
                id_territorio_corrente = territori_ridimensionati[y, x]
                if id_territorio_corrente > 0:
                    matrice_3d[y, x, 1] = label_per_id_territorio[id_territorio_corrente]

                    # Associo un identificativo ad ogni territorio F
                    if label_per_id_territorio[id_territorio_corrente] == "F":
                        matrice_3d[y, x, 1] += str(id_territorio_corrente)

                    # Seleziona il colore associato alla label riconosciuta
                    colore_scelto = COLORI_PER_LABEL[label_per_id_territorio[id_territorio_corrente]]
                    immagine_matrice_colorata[y, x] = colore_scelto
                else:
                    matrice_3d[y, x, 1] = "VUOTO"
                    immagine_matrice_colorata[y, x] = [255, 255, 255]

    # ***** SCALATURA E RICERCA START/GOAL/GATE *****
    
    # Calcoliamo di quanto abbiamo ingrandito la mappa, quindi le scale che ci servono
    altezza_originale, larghezza_originale = img_binaria.shape
    scala_x = DIMENSIONE_FINALE / larghezza_originale
    scala_y = DIMENSIONE_FINALE / altezza_originale

    # Inizializziamo il livello 0 con una stringa vuota per tenerlo pulito
    matrice_3d[:, :, 0] = ""
    trovato_c = False    # variabile per verificare che trovo C
    trovato_w = False   # variabile per verificare che trovo W
    # Scorriamo i caratteri trovati, cerchiamo C e W
    for char in caratteri_riconosciuti:
        cx_orig, cy_orig = char['centro'] # salvo coordinate baricentro carattere
        cx_scal = int(cx_orig * scala_x) # scalo coordinata x baricentro carattere
        cy_scal = int(cy_orig * scala_y) # scalo coordinata y baricentro carattere

        if matrice_3d[cy_scal, cx_scal, 1] != "CONFINE": # controllo per verificare che con la scalatura qualcosa sia finito sul confine (altamente improbabile, ma non impossibile)
            if char['label'] == 'C': 
                matrice_3d[cy_scal, cx_scal, 0] = 'C' # trovata la C salvo il carattere nelle coordinate del suo baricentro su livello 0 matrice_3d
                trovato_c = True
                cv2.circle(immagine_matrice_colorata, (cx_scal, cy_scal), 3, (0, 255, 0), -1) # metto un punto verde sulla mappa colorata sulla posizione di C

            elif char['label'] == 'W':
                matrice_3d[cy_scal, cx_scal, 0] = 'W' # trovata la W salvo il carattere nelle coordinate del suo baricentro su livello 0 matrice_3d
                trovato_w = True
                cv2.circle(immagine_matrice_colorata, (cx_scal, cy_scal), 3, (255, 0, 0), -1) # metto un punto rosso sulla mappa colorata sulla posizione di W

            elif char['label'] == 'F': # per ogni F trovata, salvo "GATE" nelle coordinate del suo baricentro su livello 0 matrice_3d
                matrice_3d[cy_scal, cx_scal, 0] = "GATE"
                 
    if not trovato_c or not trovato_w:
        print("ATTENZIONE: Commence (C) o Win (W) non trovati!")
    
    cv2.imwrite("step3_matrice_colorata.png", cv2.cvtColor(immagine_matrice_colorata, cv2.COLOR_RGB2BGR))
    print("- Salvato 'step3_matrice_colorata.png'")
    print("--- ELABORAZIONE COMPLETATA ---")

    return img_color, immagine_matrice_colorata, matrice_3d, trovato_w, trovato_c

def disegna_percorso(immagine, matrice_3d, percorso, colore): # Funzione disegna percorso algoritmo
    px, py = percorso[0]
    immagine[py, px] = colore   # Coloro punto di partenza

    for i in range(1,len(percorso)):
        x, y = percorso[i]
        dist_x = abs(x-px) # Distanza tra coordinata x attuale e precedente
        dist_y = abs(y-py) # Distanza tra coordinata y attuale e precedente
        dist_tot = dist_x + dist_y # Distanza totale
        valore_partenza = str(matrice_3d[py, px, 1])
        valore_arrivo = str(matrice_3d[y, x, 1])

        if dist_tot == 1:  # Opzione 1: passo tra due caselle 
            immagine[y, x] = colore

        elif valore_partenza.startswith("F") and valore_arrivo.startswith("F"):  # Opzione 2: salto tra due aeroporti (volo)
            cv2.circle(immagine, (px, py), 3, colore, 1)
            cv2.line(immagine, (x, y), (px, py), colore, 1)
            cv2.circle(immagine, (x, y), 3, colore, 1)

        elif (dist_x == 0 or dist_y == 0): # Opzione 3: scivolo su ghiaccio (linea dritta lunga)
            cv2.line(immagine, (x, y), (px, py), colore, 1)

        px = x # salvo pixel attuali che diventeranno i precedenti sul giro di ciclo seguente
        py = y

if __name__ == "__main__":
    nome_foto = "mappe/mappa_20.jpg"  # Controlla sempre che il nome combaci!

    img_originale, immagine_matrice_colorata, mat_3d, c, w = avvia_hub(nome_foto) #c e w bool per verificare che ho trovato c e w
    
    # ATTIVAZIONE PATHFINDING E RAPPRESENTAZIONE PERCORSO SU MAPPA
    mappa_ucs= immagine_matrice_colorata.copy()
    mappa_astar= immagine_matrice_colorata.copy()
    mappa_greedy= immagine_matrice_colorata.copy()

    if c and w:
     
        print("Colori percorsi:")
        print("UCS = verde")
        print("A* Manhattan = rosso")
        print("A* Euclideo = verde")
        print("A* OP = blu")
        print("Greedy Manhattan = rosso")
        print("Greedy Euclideo = verde")
        print("Greedy OP = blu")
        # Passiamo la matrice (contenente tutti i dati per il pathfinding) al file separato
        percorso_ucs, percorso_astar_manhattan, percorso_astar_euclideo, percorso_astar_op, percorso_greedy_manhattan, percorso_greedy_euclideo, percorso_greedy_op = pathfinding.esegui_confronto(mat_3d)

        
        # DISEGNARE IL PERCORSO SULLA MAPPA

        # Se UCS ha trovato un percorso, coloriamo i pixel del percorso di blu sulla mappa finale
        if percorso_ucs:
            disegna_percorso(mappa_ucs, mat_3d, percorso_ucs, [0, 0, 255])

        # Se A* Manhattan ha trovato un percorso, coloriamo i pixel del percorso di rosso sulla mappa finale
        if percorso_astar_manhattan:
            disegna_percorso(mappa_astar, mat_3d, percorso_astar_manhattan, [255, 0, 0])

        # Se A* Euclideo ha trovato un percorso, coloriamo i pixel del percorso di verde sulla mappa finale        
        if percorso_astar_euclideo:
            disegna_percorso(mappa_astar, mat_3d, percorso_astar_euclideo, [0, 255, 0] )
        
        # Se A* OP ha trovato un percorso, coloriamo i pixel del percorso di blu sulla mappa finale        
        if percorso_astar_op:
            disegna_percorso(mappa_astar, mat_3d, percorso_astar_op, [0, 0, 255] )

        # Se Greedy Manhattan ha trovato un percorso, coloriamo i pixel del percorso di rosso sulla mappa finale
        if percorso_greedy_manhattan:
            disegna_percorso(mappa_greedy, mat_3d, percorso_greedy_manhattan, [255, 0, 0])

        # Se Greedy Euclideo ha trovato un percorso, coloriamo i pixel del percorso di verde sulla mappa finale
        if percorso_greedy_euclideo:
            disegna_percorso(mappa_greedy, mat_3d, percorso_greedy_euclideo, [0, 255, 0])
        
        # Se Greedy OP ha trovato un percorso, coloriamo i pixel del percorso di blu sulla mappa finale        
        if percorso_greedy_op:
            disegna_percorso(mappa_greedy, mat_3d, percorso_greedy_op, [0, 0, 255] )
                
    fig, assi = plt.subplots(2, 2, figsize=(16, 10))
    assi[0,0].imshow(cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB))
    assi[0,0].set_title("1. Mappa Analizzata", fontsize=14)
    assi[0,0].axis("off")

    assi[0,1].imshow(mappa_ucs)
    assi[0,1].set_title(f"2. Mappa con UCS ({DIMENSIONE_FINALE}x{DIMENSIONE_FINALE})", fontsize=14)
    assi[0,1].axis("off")

    assi[1,0].imshow(mappa_greedy)
    assi[1,0].set_title(f"3. Mappa con Greedy con diverse euristiche ({DIMENSIONE_FINALE}x{DIMENSIONE_FINALE})", fontsize=14)
    assi[1,0].axis("off")

    assi[1,1].imshow(mappa_astar)
    assi[1,1].set_title(f"4. Mappa con A* con diverse euristiche ({DIMENSIONE_FINALE}x{DIMENSIONE_FINALE})", fontsize=14)
    assi[1,1].axis("off")


    plt.tight_layout()
    plt.show()
