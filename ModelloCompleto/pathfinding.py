import time
import math
from aima import Problem, uniform_cost_search, astar_search, greedy_best_first_graph_search


class MappaProblem(Problem):
    def __init__(self, start, goal, matrice_3d, costo_confine, costo_base, fattore_ostile):
        """
        start e goal devono essere tuple (X, Y) già scalate.
        """
        super().__init__(start, goal)
        self.matrice = matrice_3d
        self.max_y = matrice_3d.shape[0]
        self.max_x = matrice_3d.shape[1]
        self.nodi_esplorati = 0
        self.costo_confine= costo_confine
        self.costo_base= costo_base
        self.fattore_ostile= fattore_ostile

        self.aeroporti = {}    # Creo dizionario dove salvare per ogni territorio F il suo "Gate" (pixel del territorio dove si atterra in caso di volo da un altro aeroporto)
        for y in range(self.max_y):
            for x in range(self.max_x):
                valore_casella = str(self.matrice[y, x, 1])  # Lo leggo come stringa per poter usare startswith
                if valore_casella.startswith("F"):
                    if valore_casella not in self.aeroporti:  # Se non ho ancora salvato un Gate per questo aeroporto  
                        self.aeroporti[valore_casella] = (x, y) # Salvo questo pixel come suo Gate Ufficiale! (sarà il pixel più a Nord-Ovest di ogni territorio F)


    def actions(self, state):
        """
        Dato uno stato (X, Y), restituisce le mosse valide.
        """
        self.nodi_esplorati += 1 

        x, y = state
        mosse_valide = []

        # --- LOGICA VOLI AEROPORTO ---
        valore_corrente = str(self.matrice[y, x, 1])
        
        if valore_corrente.startswith("F"):
            # Scorro il dizionario dei Gate ufficiali (.items() tira fuori chiave e valore)
            for id_destinazione, (nx, ny) in self.aeroporti.items():
                # Se il Gate di destinazione è di un aeroporto diverso da quello attuale
                if id_destinazione != valore_corrente:
                    mosse_valide.append((nx, ny))
        # -----------------------------------

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
                     schianto = False
                     while(valore_pixel == "G"):
                         ny += d[1]
                         nx += d[0]
                         if 0 <= nx < self.max_x and 0 <= ny < self.max_y: #controllo che non vada fuori dalla mappa
                             valore_pixel = self.matrice[ny, nx, 1]
                         else:      #altrimenti vuol dire che si è schiantato con un bordo della mappa
                             schianto = True 
                             break
                     if valore_pixel == "X" or valore_pixel == "VUOTO" or schianto:
                         mosse_valide.append((nx-d[0], ny-d[1])) #annullo l'ultima mossa se finisco in una x, vuoto o mi sono schiantato (impossibile perché prima ci sarebbe il confine)
                     else:
                         mosse_valide.append((nx, ny))
        return mosse_valide

    def result(self, state, action):
        """
        In una griglia, l'azione coincide direttamente con il nuovo stato.
        """
        return action

    def path_cost(self, c, state1, action, state2):
        """
        Calcola il costo totale per arrivare da state1 a state2.
        """
        x, y = state2 #coordinate nodo arrivo
        px, py = state1 #coordinate nodo partenza

        # Leggo come stringhe per poter usare .startswith()
        valore_arrivo = str(self.matrice[y, x, 1])
        valore_partenza = str(self.matrice[py, px, 1])
       
        if valore_partenza.startswith("F") and valore_arrivo.startswith("F") and valore_partenza != valore_arrivo: # valore_partenza!=valore_arrivo perché sennò entrerebbe anche per gli spostamenti dentro F
            # È un volo, non si considera la distanza, il costo è fisso (il "biglietto")
            costo_movimento = self.costo_base * 3 # ho messo che il "biglietto" del volo costa 3 volte lo spostamento base (vedremo se modificarlo o anche richiederlo eventualmente)
        else:
            # È un passo normale (distanza = 1) o una scivolata su ghiaccio (distanza > 1)
            distanza = abs(x - px) + abs(y - py)
            costo_movimento = distanza * self.costo_base

        costo_aggiuntivo=0  # Eventuali costi aggiuntivi della cella di arrivo

        if valore_arrivo.isdigit():
            costo_aggiuntivo = int(valore_arrivo)* self.fattore_ostile

        elif valore_arrivo == "X":   # Non necessario in quanto già viene evitato per definizione su actions
            costo_aggiuntivo = 999999

        elif valore_arrivo == "CONFINE":
            costo_aggiuntivo = self.costo_confine
        
        return c + costo_aggiuntivo + costo_movimento

    def h1(self, node):
        """
        Euristica 1: distanza di Manhattan.
        Usata da A* Manhattan e Greedy Manhattan.
        """
        x_corrente, y_corrente = node.state
        x_goal, y_goal = self.goal

        return abs(x_corrente - x_goal) + abs(y_corrente - y_goal)

    def h2(self, node):
        """
        Euristica 2: distanza Euclidea.
        Usata da A* Euclidea.
        """
        return math.dist(node.state, self.goal)


# --- FUNZIONE PRINCIPALE ---
def esegui_confronto(matrice_3d, start_scalato, goal_scalato):
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

    problema = MappaProblem(start_scalato, goal_scalato, matrice_3d, confine, base, ostile)

    # 1. Uniform Cost Search
    print("\nAvvio Ricerca NON Informata: Uniform-Cost Search...")
    start_time = time.time()

    nodo_ucs = uniform_cost_search(problema)

    tempo_ucs = time.time() - start_time
    numero_nodi_ucs = problema.nodi_esplorati

    if nodo_ucs:
        print("Percorso UCS trovato!")
        print(f" -> Costo: {nodo_ucs.path_cost}")
        print(f" -> Tempo: {tempo_ucs:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_ucs:,}")
        percorso_ucs = nodo_ucs.solution()
    else:
        print("UCS fallita: percorso non trovato.")
        percorso_ucs = None

    problema.nodi_esplorati = 0

    # 2. A* con Manhattan
    print("\nAvvio Ricerca Informata: A* con distanza di Manhattan...")
    start_time = time.time()

    nodo_astar1 = astar_search(problema, problema.h1)

    tempo_astar1 = time.time() - start_time
    numero_nodi_astar1 = problema.nodi_esplorati

    if nodo_astar1:
        print("Percorso A* Manhattan trovato!")
        print(f" -> Costo: {nodo_astar1.path_cost}")
        print(f" -> Tempo: {tempo_astar1:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_astar1:,}")
        percorso_astar1 = nodo_astar1.solution()
    else:
        print("A* Manhattan fallito: percorso non trovato.")
        percorso_astar1 = None

    problema.nodi_esplorati = 0

    # 3. A* con Euclidea
    print("\nAvvio Ricerca Informata: A* con distanza Euclidea...")
    start_time = time.time()

    nodo_astar2 = astar_search(problema, problema.h2)

    tempo_astar2 = time.time() - start_time
    numero_nodi_astar2 = problema.nodi_esplorati

    if nodo_astar2:
        print("Percorso A* Euclidea trovato!")
        print(f" -> Costo: {nodo_astar2.path_cost}")
        print(f" -> Tempo: {tempo_astar2:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_astar2:,}")
        percorso_astar2 = nodo_astar2.solution()
    else:
        print("A* Euclidea fallito: percorso non trovato.")
        percorso_astar2 = None

    problema.nodi_esplorati = 0

    # 4. Greedy Best First Search con Manhattan
    print("\nAvvio Ricerca Informata: Greedy Best First Search con Manhattan...")
    start_time = time.time()

    nodo_greedy1 = greedy_best_first_graph_search(problema, problema.h1)

    tempo_greedy1 = time.time() - start_time
    numero_nodi_greedy1 = problema.nodi_esplorati

    if nodo_greedy1:
        print("Percorso Greedy Manhattan trovato!")
        print(f" -> Costo: {nodo_greedy1.path_cost}")
        print(f" -> Tempo: {tempo_greedy1:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_greedy1:,}")
        percorso_greedy1 = nodo_greedy1.solution()
    else:
        print("Greedy Best First Search Manhattan fallito: percorso non trovato.")
        percorso_greedy1 = None

    problema.nodi_esplorati = 0

    # 5. Greedy Best First Search con Euclidea
    print("\nAvvio Ricerca Informata: Greedy Best First Search con distanza Euclidea...")
    start_time = time.time()

    nodo_greedy2 = greedy_best_first_graph_search(problema, problema.h2)

    tempo_greedy2 = time.time() - start_time
    numero_nodi_greedy2 = problema.nodi_esplorati

    if nodo_greedy2:
        print("Percorso Greedy Euclidea trovato!")
        print(f" -> Costo: {nodo_greedy2.path_cost}")
        print(f" -> Tempo: {tempo_greedy2:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_greedy2:,}")
        percorso_greedy2 = nodo_greedy2.solution()
    else:
        print("Greedy Best First Search Euclidea fallito: percorso non trovato.")
        percorso_greedy2 = None

    return percorso_ucs, percorso_astar1, percorso_astar2, percorso_greedy1, percorso_greedy2
