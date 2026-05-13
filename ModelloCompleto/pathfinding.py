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


    def actions(self, state):
        """
        Dato uno stato (X, Y), restituisce le mosse valide.
        """
        self.nodi_esplorati += 1

        x, y = state
        mosse_valide = []

        # Movimento in 4 direzioni: su, giù, sinistra, destra
        direzioni = [
            (x, y - 1),
            (x, y + 1),
            (x - 1, y),
            (x + 1, y)
        ]

        for nx, ny in direzioni:
            # Controllo che il vicino sia dentro la mappa
            if 0 <= nx < self.max_x and 0 <= ny < self.max_y:

                valore_pixel = self.matrice[ny, nx, 1]

                # Accetto solo celle che non siano ostacoli o vuoto/oceano
                if valore_pixel != "X" and valore_pixel != "VUOTO":
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
        x, y = state2

        valore_pixel = self.matrice[y, x, 1]

        costo_aggiuntivo=0

        if valore_pixel.isdigit():
            costo_aggiuntivo = int(valore_pixel)* self.fattore_ostile

        elif valore_pixel == "X":
            costo_aggiuntivo = 999999

        elif valore_pixel == "CONFINE":
            costo_aggiuntivo = self.costo_confine

        return c + costo_aggiuntivo + self.costo_base

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