# Definizione del Dominio: Navigazione e Mappatura Territoriale

Il presente progetto si pone l'obiettivo di modellare un sistema di navigazione per un agente operante in un ambiente discretizzato, caratterizzato da territori frazionati e costi di transito variabili.

## 1. Architettura della Mappa e Discretizzazione

L'ambiente è rappresentato come una **matrice di celle**, dove la granularità della griglia determina la precisione della rappresentazione geografica:

* **Rappresentazione degli Stati**: Uno stato (o regione) può coincidere con una singola cella o essere costituito da un cluster di più celle contigue per rappresentare estensioni territoriali maggiori.
* **Segmentazione Cromatica**: Al fine di distinguere topologicamente le diverse regioni, ogni stato è caratterizzato da un colore differente rispetto a quelli adiacenti. Questo garantisce una separazione netta dei confini durante la fase di analisi dell'immagine.
* **Processo di Pixellazione**: La definizione dei confini avviene attraverso una procedura di pixellazione che mappa l'input visivo sulla griglia di navigazione, definendo l'appartenenza di ogni coordinata $(x, y)$ a uno specifico stato.

## 2. Logica di Movimento e Modello dei Costi

Il movimento dell'agente è vincolato a spostamenti tra celle adiacenti (N, S, E, W). La funzione di costo associata alle azioni di navigazione è così definita:

* **Costo Base di Spostamento**: Il transito tra due celle adiacenti ha un costo base unitario pari a **1**.
* **Penalità di Confine**: L'attraversamento di un confine tra due stati distinti comporta un onere aggiuntivo di **+1** sul costo totale dello spostamento (quindi un totale di 2).
* **Penalità Territorio Nemico**: L'ingresso in uno stato ostile applica un malus aggiuntivo istantaneo al costo del tragitto. Tale malus è calcolato moltiplicando il livello di pericolosità del territorio nemico (da 1 a 9) per un parametro di calibrazione $X$ (da definire in fase di configurazione).

### Formalizzazione Matematica
Dato un percorso che attraversa $n$ celle, interseca $k$ confini di stato e prevede l'ingresso in $m$ stati nemici (ciascuno caratterizzato da un livello di pericolosità $L_i$), il costo totale $C$ è espresso dalla formula:
$$C = n + k + \sum_{i=1}^{m} (L_i \cdot X)$$

*Esempio pratico*: Uno spostamento attraverso 6 celle all'interno dello stesso stato alleato ha un costo di **6**. Se lo stesso spostamento prevede l'attraversamento di un confine per entrare in uno stato nemico con livello di pericolosità 4, il calcolo del costo diverrà: 6 (costo celle) + 1 (costo confine) + $4 \cdot X$ (malus nemico). Il costo totale sarà quindi pari a **$7 + 4X$**.

## 3. Classificazione e Tipologia dei Territori

In fase di acquisizione, ogni cella viene analizzata e classificata per definirne le proprietà fisiche e tattiche. Invece di utilizzare nomi estesi, il sistema adotta una codifica alfanumerica:

| Identificatore | Tipologia | Proprietà |
| :--- | :--- | :--- |
| **S** | **Start** | Punto di origine dell'agente. |
| **G** | **Goal** | Obiettivo finale o destinazione del percorso. |
| **A** | **Alleato** | Territorio amichevole con transito standard. |
| **X** | **Confine Naturale** | Ostacolo insormontabile (non attraversabile). |
| **1 - 9** | **Territorio Nemico** | Rappresenta il grado di pericolosità crescente. |

I valori numerici relativi alla pericolosità dei territori nemici sono direttamente integrati nella funzione di costo. Questo approccio parametrico (modulabile tramite la variabile $X$) permette all'algoritmo di ricerca di valutare percorsi più o meno prudenti, bilanciando in modo dinamico la brevità del tragitto con l'esposizione al rischio.
