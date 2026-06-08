import time
import math
from aima import Problem, uniform_cost_search, astar_search, greedy_best_first_graph_search


class MappaProblem(Problem):
    def __init__(self, matrice_3d, costo_confine, costo_base, fattore_ostile, costo_volo):
        # inizializzo variabili problema
        self.matrice = matrice_3d
        self.max_y = matrice_3d.shape[0]
        self.max_x = matrice_3d.shape[1]
        self.nodi_esplorati = 0
        self.costo_confine = costo_confine
        self.costo_base = costo_base
        self.fattore_ostile = fattore_ostile
        self.costo_biglietto = costo_base * costo_volo

        start = None
        goal = None
        self.aeroporti = {}    # Dizionario dove salvare, per ogni territorio F, il suo "Gate"

        # Ciclo per mappare il mondo (trova start, goal, gate)
        for y in range(self.max_y):
            for x in range(self.max_x):
                valore_casella = str(self.matrice[y, x, 1])  # Lo leggo come stringa per poter usare startswith
                trova_baricentri = self.matrice[y, x, 0]
                if trova_baricentri == "C":
                    start = (x, y)
                elif trova_baricentri == "W":
                    goal = (x, y) 
                elif valore_casella.startswith("F") and trova_baricentri == "GATE":
                    self.aeroporti[valore_casella] = (x, y) # Salvo questo pixel come suo gate ufficiale
        super().__init__(start, goal)


    def actions(self, state):
        # Dato un pizel (x, y), restituisce le mosse valide.
        self.nodi_esplorati += 1 

        x, y = state
        mosse_valide = []

        # *** VOLI AEROPORTO ***
        valore_corrente = str(self.matrice[y, x, 1])
        # Controllo se sono in un territorio F e se le mie coordinate (x,y) corrispondono a quelle del gate ufficiale di questo aeroporto.
        if valore_corrente.startswith("F") and (x, y) == self.aeroporti.get(valore_corrente):
            # Scorro il dizionario dei Gate ufficiali (.items() tira fuori chiave e valore)
            for id_destinazione, (nx, ny) in self.aeroporti.items():
                # Se il Gate di destinazione è di un aeroporto diverso da quello attuale
                if id_destinazione != valore_corrente:
                    mosse_valide.append((nx, ny))
        # ***********************************

        direzioni = [
            (0, -1), # Nord
            (1, 0), # Est
            (-1, 0), # Ovest
            (0, 1) # Sud
        ]
        
        for d in direzioni:
            ny = y + d[1]
            nx = x + d[0]
            if 0 <= nx < self.max_x and 0 <= ny < self.max_y:
                valore_pixel = self.matrice[ny, nx, 1]
                if valore_pixel != "X" and valore_pixel != "VUOTO" and valore_pixel != "G":
                    mosse_valide.append((nx, ny))
                elif valore_pixel == "G":
                    while valore_pixel == "G":
                        ny += d[1]
                        nx += d[0]
                        valore_pixel = self.matrice[ny, nx, 1]
                    mosse_valide.append((nx, ny))
        return mosse_valide

    def result(self, state, action):
        # In una griglia, l'azione coincide direttamente con il nuovo stato.
        return action

    def path_cost(self, c, state1, action, state2):
        # Calcola il costo totale per arrivare da state1 a state2.
        x, y = state2 #coordinate nodo arrivo
        px, py = state1 #coordinate nodo partenza

        # Leggo come stringhe per poter usare .startswith()
        valore_arrivo = str(self.matrice[y, x, 1])
        valore_partenza = str(self.matrice[py, px, 1])
       
        if valore_partenza.startswith("F") and valore_arrivo.startswith("F") and valore_partenza != valore_arrivo: # valore_partenza!=valore_arrivo perché sennò entrerebbe anche per gli spostamenti dentro F
            # È un volo, non si considera la distanza, il costo è fisso (il "biglietto")
            costo_movimento = self.costo_biglietto
        else:
            # È un passo normale (distanza = 1) o una scivolata su ghiaccio (distanza > 1)
            distanza = abs(x - px) + abs(y - py)
            costo_movimento = distanza * self.costo_base

        costo_aggiuntivo=0  # Eventuali costi aggiuntivi della cella di arrivo

        if valore_arrivo.isdigit():
            costo_aggiuntivo = int(valore_arrivo)* self.fattore_ostile

        elif valore_arrivo == "CONFINE":
            costo_aggiuntivo = self.costo_confine
        
        return c + costo_aggiuntivo + costo_movimento

    def h1(self, node):
        # Euristica 1: distanza di Manhattan.
        # Usata da A* Manhattan e Greedy Manhattan.
        x, y = node.state
        gx, gy = self.goal

        return (abs(x - gx) + abs(y - gy)) * self.costo_base

    def h2(self, node):
        # Euristica 2: distanza Euclidea.
        # Usata da A* Euclidea e Greedy Euclideo.
        return math.dist(node.state, self.goal) * self.costo_base
    
    def h3(self, node):
        # Euristica 3: Euristica OP (Manhattan personalizzata), mantiene ammissibilità anche con aeroporti
        x, y = node.state
        gx, gy = self.goal

        costo_camminata = (abs(x - gx) + abs(y - gy)) * self.costo_base
        if len(self.aeroporti) < 2: # Se non ci sono almeno 2 aeroporti, il problema del volo non si pone, si restituisce la distanza di Manhattan (calcolata considerando il costo base)
            return costo_camminata

        # Distanza di manhattan tra il pixel attuale e l'aeroporto più vicino
        distanza_attuale_aeroporto = min(abs(x - x_aeroporto) + abs(y - y_aeroporto) for x_aeroporto, y_aeroporto in self.aeroporti.values())
        # Distanza di manhattan tra il goal e l'aeroporto più vicino ad esso
        distanza_aeroporto_goal = min(abs(x_aeroporto - gx) + abs(y_aeroporto - gy) for x_aeroporto, y_aeroporto in self.aeroporti.values())

        # se si sceglie di prendere il volo il costo è la somma delle due distanze calcolate in precedenza (moltiplicate per costo base) + il costo del biglietto
        costo_volo = (distanza_attuale_aeroporto + distanza_aeroporto_goal) * self.costo_base + self.costo_biglietto

        return min(costo_camminata, costo_volo)

def esegui_ricerca(nome, algoritmo, problema, euristica=None):
    # Esegue un algoritmo di ricerca, fornisce i tempi, stampa i dati
    # e restituisce il percorso trovato.

    problema.nodi_esplorati = 0 # reset contatore nodi esplorati

    print(f"\nAvvio ricerca: {nome}")
    start_time = time.time()

    if euristica: # se c'è l'euristica la passo sennò no
        nodo_finale = algoritmo(problema, euristica)
    else:
        nodo_finale = algoritmo(problema)

    tempo_impiegato = time.time() - start_time
    nodi_esplorati = problema.nodi_esplorati
    
    # Stampa dei risultati
    if nodo_finale:
        print(f"Percorso {nome} trovato")
        print(f" -> Costo: {nodo_finale.path_cost}")
        print(f" -> Tempo: {tempo_impiegato:.2f} secondi")
        print(f" -> Nodi Esplorati: {nodi_esplorati:,}")
        percorso = nodo_finale.solution()
    else:
        print(f"{nome} fallito: percorso non trovato.")
        percorso = None
        
    problema.nodi_esplorati = 0 # reset di sicurezza finale
    
    return percorso

# *** FUNZIONE PRINCIPALE ***
def esegui_confronto(matrice_3d):
    print("\nInizializzazione Problema AIMA...")

    while True:
        try:
            base = float(input("Quanto costa uno spostamento base tra una cella e l'altra? ").replace(",", "."))
        except ValueError:
            print("Costo non valido")
            continue

        if base < 1:
            print("Costo non valido, minimo 1")
            continue
        else:
            break

    while True:
        try:
            confine = float(input("Quanto costa uno spostamento nel confine? In aggiunta a quello base: ").replace(",", "."))
        except ValueError:
            print("Costo non valido")
            continue

        if confine < 0:
            print("Costo non valido, minimo 0")
            continue
        else:
            break

    while True:
        try:
            ostile = float(input("Per quanto moltiplicare l'ostilità dei nemici già presenti nella mappa? Valore da 1 a 10: ").replace(",", "."))
        except ValueError:
            print("Costo non valido")
            continue

        if ostile < 1 or ostile > 10:
            print("Costo non valido, minimo 1 e massimo 10")
            continue
        else:
            break

    while True:
        try:
            costo_volo = float(input("Quanto costa usare l'aereo? Il fattore verrà moltiplicato per il costo base: ").replace(",", "."))
        except ValueError:
            print("Costo non valido")
            continue

        if costo_volo < 1:
            print("Costo non valido, minimo 1")
            continue
        else:
            break


    problema = MappaProblem(matrice_3d, confine, base, ostile, costo_volo)

    # *** ESECUZIONE ALGORITMI ***

    # Uniform Cost Search (Senza euristica!)
    percorso_ucs = esegui_ricerca("Uniform Cost Search", uniform_cost_search, problema) 

    # A* con Manhattan
    percorso_astar1 = esegui_ricerca("A* Manhattan", astar_search, problema, problema.h1)

    # A* con Euclidea
    percorso_astar2 = esegui_ricerca("A* Euclidea", astar_search, problema, problema.h2) 

    # A* con euristica OP
    percorso_astar3 = esegui_ricerca("A* OP", astar_search, problema, problema.h3) 

    # Greedy con Manhattan
    percorso_greedy1 = esegui_ricerca("Greedy Manhattan", greedy_best_first_graph_search, problema, problema.h1)

    # Greedy con Euclidea
    percorso_greedy2 = esegui_ricerca("Greedy Euclideo", greedy_best_first_graph_search, problema, problema.h2)   

    # Greedy con euristica OP
    percorso_greedy3 = esegui_ricerca("Greedy OP", greedy_best_first_graph_search, problema, problema.h3)
    
    
    return percorso_ucs, percorso_astar1, percorso_astar2, percorso_astar3, percorso_greedy1, percorso_greedy2, percorso_greedy3
