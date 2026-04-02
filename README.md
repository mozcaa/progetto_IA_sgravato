# progetto_IA_sgravato
# Analisi del Dominio: Formalizzazione del Movimento e dei Costi

[cite_start]Seguendo le specifiche di progetto per il dominio *Uniform Coloring*[cite: 6], la descrizione formale delle regole di movimento e della struttura della griglia è definita come segue:

## 1. Struttura della Griglia e Definizione di Stato
[cite_start]L'ambiente operativo è modellato come una griglia rettangolare discreta[cite: 8], dove l'unità minima di spazio è rappresentata dalla cella.
* [cite_start]**Composizione degli Stati**: Uno "stato" (o regione) è costituito da un insieme di più celle contigue[cite: 1].
* [cite_start]**Risoluzione dei Confini**: La delimitazione dei confini tra i diversi stati viene determinata tramite un processo di "pixellazione", volto a definire l'appartenenza delle celle a una specifica regione in base alla dimensione della griglia impostata[cite: 1].

## 2. Funzione di Costo dello Spostamento
[cite_start]L'agente (la testina colorante $T$) può muoversi tra celle attigue nelle quattro direzioni cardinali (N, S, E, W)[cite: 8]. Il costo associato a tali azioni non è puramente uniforme, ma dipende dalla transizione tra le regioni:

* [cite_start]**Costo Base della Cella**: Ogni cella all'interno della griglia ha un costo di percorrenza intrinseco pari a $1$ ($cost_{cella} = 1$)[cite: 1].
* [cite_start]**Penalità di Transizione di Stato**: Ogni cambiamento di stato, inteso come l'attraversamento di un confine tra due diverse regioni, comporta un incremento del costo di spostamento pari a $+1$[cite: 2].

### Formalizzazione del Calcolo
Dato un percorso composto da $n$ celle, il costo totale $C$ è espresso dalla formula:
$$C = n + k$$
Dove:
* [cite_start]$n$ è il numero di celle percorse (costo base)[cite: 1].
* [cite_start]$k$ è il numero di confini di stato attraversati durante il movimento[cite: 2].

**Esempi Applicativi:**
1. [cite_start]**Movimento Intra-Stato**: Se l'agente si sposta attraverso $6$ celle appartenenti alla medesima regione, il costo totale sarà pari a $6$[cite: 2].
2. [cite_start]**Movimento Inter-Stato**: Se l'agente percorre $6$ celle ma attraversa un confine di stato, il costo totale risulterà pari a $7$ (costo delle celle + penalità di confine)[cite: 2].

## 3. Vincoli del Movimento
[cite_start]In linea con i vincoli del dominio[cite: 14]:
* [cite_start]**$v_1$**: L'agente può compiere un solo passo alla volta[cite: 14].
* **$v_2$**: Lo spostamento è consentito esclusivamente tra celle adiacenti[cite: 14].
* **$v_3$**: Non è permesso all'agente di uscire dai confini della griglia definiti in fase di acquisizione[cite: 8].
