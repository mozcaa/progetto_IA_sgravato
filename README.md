# Analisi del Dominio: Formalizzazione del Movimento e dei Costi

## 1. Struttura della Griglia e Definizione di Stato
L'ambiente operativo è modellato come una griglia rettangolare discreta, dove l'unità minima di spazio è rappresentata dalla cella.
* **Composizione degli Stati**: Uno "stato" (o regione) è costituito da un insieme di più celle contigue.
* **Risoluzione dei Confini**: La delimitazione dei confini tra i diversi stati viene determinata tramite un processo di "pixellazione", volto a definire l'appartenenza delle celle a una specifica regione in base alla dimensione della griglia impostata.

## 2. Funzione di Costo dello Spostamento
L'agente (la testina colorante $T$) può muoversi tra celle attigue nelle quattro direzioni cardinali (N, S, E, W). Il costo associato a tali azioni non è puramente uniforme, ma dipende dalla transizione tra le regioni:

* **Costo Base della Cella**: Ogni cella all'interno della griglia ha un costo di percorrenza intrinseco pari a $1$ ($cost_{cella} = 1$).
* **Penalità di Transizione di Stato**: Ogni cambiamento di stato, inteso come l'attraversamento di un confine tra due diverse regioni, comporta un incremento del costo di spostamento pari a $+1$.

### Formalizzazione del Calcolo
Dato un percorso composto da $n$ celle, il costo totale $C$ è espresso dalla formula:
$$C = n + k$$
Dove:
* $n$ è il numero di celle percorse (costo base).
* $k$ è il numero di confini di stato attraversati durante il movimento.

**Esempi Applicativi:**
1. **Movimento Intra-Stato**: Se l'agente si sposta attraverso $6$ celle appartenenti alla medesima regione, il costo totale sarà pari a $6$.
2. **Movimento Inter-Stato**: Se l'agente percorre $6$ celle ma attraversa un confine di stato, il costo totale risulterà pari a $7$ (costo delle celle + penalità di confine).

## 3. Vincoli del Movimento
In linea con i vincoli del dominio:
* **$v_1$**: L'agente può compiere un solo passo alla volta.
* **$v_2$**: Lo spostamento è consentito esclusivamente tra celle adiacenti.
* **$v_3$**: Non è permesso all'agente di uscire dai confini della griglia definiti in fase di acquisizione.
