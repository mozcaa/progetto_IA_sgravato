import time
import math
from aima import Problem, uniform_cost_search, astar_search

class MappaProblem(Problem):
    def __init__(self, start, goal, matrice_3d):
        """
        start e goal devono essere tuple (X, Y) già scalate.
        """
        super().__init__(start, goal)
        self.matrice = matrice_3d
        self.max_y = matrice_3d.shape[0]
        self.max_x = matrice_3d.shape[1]
        self.nodi_esplorati = 0 #contatore nodi

    # def goal_test(self, state):
    #     """
    #     Invece di cercare la coordinata esatta, vince appena calpesta
    #     un pixel che appartiene al territorio "W".
    #     """
    #     x, y = state
        
    #     # Controllo se nel Livello 1 di questo pixel c'è la stringa "W"
    #     if self.matrice[y, x, 1] == "W":
    #         return True
            
    #     return False
    
    def actions(self, state):
        """
        Dato uno stato (X, Y), restituisce le mosse valide (vicini).
        """
        self.nodi_esplorati += 1 #incremento contatore
        x, y = state
        mosse_valide = []
        
        # Le 4 direzioni: Su, Giù, Sinistra, Destra
        direzioni = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        
        for nx, ny in direzioni:
            # 1. Controllo se sono dentro i bordi della mappa
            if 0 <= nx < self.max_x and 0 <= ny < self.max_y:
                # 2. Accetta il pixel SOLO se la sua label NON è "X" o "VUOTO"(oceano)
                if self.matrice[ny, nx, 1] != "X" and self.matrice[ny, nx, 1] != "VUOTO": 
                    mosse_valide.append((nx, ny))
                    
        return mosse_valide

    def result(self, state, action):
        """
        In una griglia, l'azione (spostarsi in un pixel) coincide con il nuovo stato.
        """
        return action

    def path_cost(self, c, state1, action, state2):
        """
        Calcola il costo per muoversi da state1 a state2.
        """
        x, y = state2
        valore_pixel = self.matrice[y, x, 1] # Estraggo la stringa es. "4", "A", "VUOTO"
        tipo_pixel = self.matrice[y, x, 0]
        
        # Definiamo i costi in base a cosa c'è scritto nel pixel
        costo_aggiuntivo = 1 # Costo base
        
        if valore_pixel.isdigit():
            costo_aggiuntivo = int(valore_pixel) # Es. se è "4", il costo è 4
        elif valore_pixel == "X":
            costo_aggiuntivo = 999999 # Costo altissimo (ostacolo)
        # elif valore_pixel == "CONFINE":
        #     costo_aggiuntivo = 10
        # Puoi aggiungere altre regole per "A", "VUOTO", ecc.
            
        return c + costo_aggiuntivo
    
    def h(self, node):
        """
        Calcola l'euristica (Distanza di Manhattan).
        Essendo un metodo della classe, ha pieno accesso a self.goal!
        """
        x_corrente, y_corrente = node.state
        x_goal, y_goal = self.goal
        
        return abs(x_corrente - x_goal) + abs(y_corrente - y_goal)

# --- FUNZIONE PRINCIPALE ---
def esegui_confronto(matrice_3d, start_scalato, goal_scalato):
    print("\nInizializzazione Problema AIMA...")
    problema = MappaProblem(start_scalato, goal_scalato, matrice_3d)

    # 1. Ricerca NON informata (Uniform-Cost-Search)
    print("\nAvvio Ricerca NON Informata (Uniform-Cost Search)...")
    start_time = time.time()
    nodo_ucs = uniform_cost_search(problema)
    tempo_ucs = time.time() - start_time
    numero_nodi_ucs = problema.nodi_esplorati
    
    if nodo_ucs:
        print(f"Percorso UCS trovato!")
        print(f" -> Costo: {nodo_ucs.path_cost}")
        print(f" -> Tempo: {tempo_ucs:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_ucs:,}") # Il ':, ' aggiunge i separatori delle migliaia
        percorso_ucs = nodo_ucs.solution() # Restituisce la lista di (X, Y)
    else:
        print("UCS fallita: percorso non trovato.")
        percorso_ucs = None #se il percorso non viene trovato, così return finale non si bugga

    problema.nodi_esplorati = 0 #azzero contatore nodi

    # 2. Ricerca Informata (A-Star)
    print("\nAvvio Ricerca Informata (A* Search)...")
    start_time = time.time()
    nodo_astar = astar_search(problema)
    tempo_astar = time.time() - start_time
    numero_nodi_astar = problema.nodi_esplorati

    if nodo_astar:
        print(f"Percorso A* trovato!")
        print(f" -> Costo: {nodo_astar.path_cost}")
        print(f" -> Tempo: {tempo_astar:.2f} secondi")
        print(f" -> Nodi Esplorati: {numero_nodi_astar:,}")
        percorso_astar = nodo_astar.solution() # Restituisce la lista di (X, Y)
    else:
        print("A* fallito: percorso non trovato.")
        percorso_astar = None #se il percorso non viene trovato, così return finale non si bugga

    return percorso_ucs, percorso_astar