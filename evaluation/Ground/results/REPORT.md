# Ground: prima valutazione dell'activity recognition

Input: `ground_action_recognition_v1.zip`, 80 JSON validi; esclusi i metadati
macOS. Riferimento: 247 segmenti in `../ground_truth_actions.csv`.

## Metodo e limiti

La corrispondenza delle identità è stimata scegliendo, per ciascuna clip,
l'abbinamento uno-a-uno che massimizza i frame classificati correttamente.
È una valutazione **ottimistica con identità inferite dalle azioni**, non una
misura indipendente del riconoscimento o della correttezza del tracking.
Si ottimizza l'accuracy; gli F1 sono calcolati sul medesimo abbinamento.
L'identità rimane fissa per tutta la clip, senza cambi per finestra o frame.
Eventuali track aggiuntivi esclusi sono elencati nel mapping; non sono penalizzati
come falsi positivi di detection, perché questa è una valutazione sulle persone GT.

Si assume che scene1 inizi al frame 0 della registrazione annotata e che ogni
part inizi a `(part-1)*300`, coerentemente con il preprocessing a 30 FPS e blocchi
da 10 secondi. Start GT incluso, End GT incluso; le finestre predette hanno End
escluso. Le durate sono compatibili, ma il frame iniziale esatto non è verificato
sui video originali. Non sono stati ottimizzati offset o velocità contro il GT.

Per finestre sovrapposte si sceglie la predizione con centro più vicino al frame;
in caso di pari distanza vince la finestra precedente. Si valutano tutti i frame
annotati, contando le lacune come `no_prediction`. Non sono metriche di detection
di eventi: nessuna soglia IoU temporale è applicata.

## Identità inferite

| Video | Corrispondenza prevalente |
|---|---|
| P000103_T1_V1 | B → 1001; C → 1000 |
| P000103_T1_V2 | B → 1001; C → 1000 |
| P000200_T1_V1 | B → 1000, unico track e unica persona annotata |
| P000301_T1_V1 | B → 1001; C → 1000 |
| P010203_T1_V1 | A → 1001; B → 1002; C → 1000 |

Eccezione: P000103_T1_V1, part6: B → 1000; C → 1003; track 1001 escluso.
Nelle ultime clip P000103_T1_V1 part9 e P000103_T1_V2 part8 i due abbinamenti
sono equivalenti: tutte le predizioni sono Idle, quindi l'identità non è deducibile.
La scelta deterministica in quei casi non cambia le metriche. Alcune altre clip
hanno un margine ridotto: il file `person_mapping.csv` riporta differenze in frame
tra le prime due soluzioni; non si tratta di una probabilità o confidenza calibrata.
`mapping_alternatives.csv` conserva tutte le alternative valutate.

## Risultati per frame-persona

| Metrica | Valore |
|---|---:|
| Accuracy | 52,83% |
| Macro-F1, tutte le 12 classi | 33,14% |
| F1 pesato per supporto | 47,45% |
| Frame-persona annotati | 22.406 |
| Unknown | 0,00% |
| Nessuna predizione | 0,27% (60 frame-persona) |

| Azione | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cut | 26,06% | 22,34% | 24,06% |
| Drop | 0,00% | 0,00% | 0,00% |
| Grasp | 65,19% | 4,33% | 8,13% |
| Handover | 64,44% | 10,49% | 18,04% |
| Hold | 43,79% | 74,67% | 55,21% |
| Idle | 82,41% | 80,01% | 81,20% |
| Place down | 45,03% | 18,95% | 26,68% |
| Pour | 56,54% | 87,73% | 68,76% |
| Push | 65,56% | 51,75% | 57,84% |
| Shake | 25,45% | 86,28% | 39,31% |
| Squash | 0,00% | 0,00% | 0,00% |
| Twist | 47,59% | 11,48% | 18,50% |

Precision indefinita per classi mai predette è riportata a zero. `Drop` non viene
mai predetto nell'archivio; `Squash` viene predetto ma non coincide con i relativi
frame GT nell'abbinamento scelto. `Shake` è sovrapredetto; Grasp, Handover,
Place_Down e Twist hanno recall basso. La finestra di 2 secondi con un solo verbo
dominante può penalizzare le transizioni brevi; questi risultati non dimostrano
che sia l'unica causa degli errori. Unknown a zero significa assenza di astensioni,
non assenza di ambiguità.

## Riproduzione

### Sequenza completa delle azioni

Il notebook include anche una sezione trace. Una trace corrisponde a una persona
GT in un intero video originale: 10 trace nei cinque video. Le clip sono unite
con lo stesso mapping usato per le metriche per frame; il mapping non è
riottimizzato per migliorare le sequenze. Nessun ordine artificiale è imposto
fra azioni simultanee di persone differenti. La ricostruzione rimane ottimistica
e soggetta a possibili cambi di identità ai confini clip.

Si comprimono le ripetizioni consecutive di GT e predizioni dopo l'allineamento
per frame e l'unione delle clip. Non si filtrano eventi brevi. Unknown e
no_prediction restano eventi che possono introdurre errori. Eventuali buchi
interni nel GT fanno interrompere il valutatore, evitando concatenazioni implicite.

| Sottoinsieme della sequenza | Trace esatte | Edit similarity media | LCS-F1 medio |
|---|---:|---:|---:|
| Tutte le azioni | 0/10 | 35,49% | 44,59% |
| Senza Drop | 0/10 | 36,43% | 46,10% |
| Senza Idle | 0/10 | 29,63% | 38,37% |
| Senza Idle e Drop | 0/10 | 31,03% | 40,22% |
| Senza Idle e Hold | 0/10 | 25,94% | 37,67% |
| Senza Idle, Hold e Drop | 0/10 | 27,97% | 40,56% |

Edit similarity = 1 − distanza Levenshtein / massima lunghezza delle due trace;
inserimento, cancellazione e sostituzione costano 1. LCS è la sottosequenza comune
più lunga che preserva l'ordine: precision = LCS / eventi predetti, recall = LCS /
eventi GT, F1 = 2 LCS / somma delle lunghezze. Due trace vuote valgono 1;
una sola vuota vale 0 per le similarità. Le medie danno uguale peso alle trace.
La distanza di modifica media completa è 19 operazioni per trace.

I risultati sono in `per_trace.csv` (sequenze incluse) e `trace_averages.csv`.
Le metriche di sequenza ignorano durate e precisione dei confini: affiancarle a
quelle per frame. Due segmenti adiacenti con la stessa etichetta vengono fusi
anche se nel log sono righe distinte: qui si valuta la sequenza dei soli verbi.
Nelle varianti senza classi si filtrano le etichette da ENTRAMBE le sequenze e
poi si comprime nuovamente: è una proiezione della trace, diversa dall'esclusione
di classi dalle medie per frame che conserva tutti i falsi positivi.

Notebook autonomo per Colab o Jupyter: `../activity_recognition_evaluation.ipynb`.
Include il valutatore e i 247 segmenti GT: basta indicare lo ZIP delle predizioni.
In Colab legge per default lo ZIP da DATA_ROOT su Drive e salva i risultati nella
sottocartella `ground_activity_evaluation_v1`, con archivio scaricabile finale.
Non richiede GPU. I dati GT incorporati sono una copia esplicita: aggiornarli o
impostare GT_CSV quando si valutano altre annotazioni.

Escludendo solo Drop dalle classi mediate, Macro-F1 sale da 33,14% a 36,16%,
Micro-F1 da 52,90% a 53,61% e F1 pesato da 47,45% a 48,74%.
L'accuracy completa resta 52,83%. Non vengono rimossi i 592 frame GT Drop:
gli errori commessi su quei frame rimangono FP delle altre azioni. L'esclusione
riduce le classi della media, non migliora le predizioni. Le medie complete
restano il risultato principale; NO_DROP è un'analisi aggiuntiva post hoc.

Il riepilogo `evaluation_results.txt` usa le metriche applicabili dell'esempio
TUMKitchen fornito: Precision/Recall/F1 per classe con supporto, TP/FP/FN e
coverage; medie macro e micro; sono aggiunte anche le medie pesate. `averages.csv`
contiene le medie, `per_class.csv` le singole azioni e `per_video.csv` i video.

| Classi considerate nella media | Macro-P | Macro-R | Macro-F1 | Micro-F1 | F1 pesato |
|---|---:|---:|---:|---:|---:|
| Tutte le 12 | 43,51% | 37,34% | 33,14% | 52,90% | 47,45% |
| Senza Idle | 39,97% | 33,46% | 28,77% | 41,59% | 33,70% |
| Senza Idle e Hold | 39,59% | 29,34% | 26,13% | 34,71% | 26,57% |

Macro-F1 è la media degli F1 per classe, non l'F1 ottenuto dalle medie di
precision e recall. Le classi escluse non partecipano alla media; i loro frame
non vengono eliminati dalla matrice: una predizione Grasp su GT Idle rimane un
FP di Grasp. Le persone non vengono riabbinate cambiando il sottoinsieme di classi.

Accuracy media per video: 51,36%; copertura media per video: 99,75%.
Queste medie assegnano uguale peso ai cinque video originali, non alle 80 coppie
clip-track. La copertura pooled è 99,73%. Per azione, coverage è la quota dei
frame GT di quell'azione che hanno una predizione, anche se errata o Unknown.
Unknown e no_prediction causano FN per la classe GT e non FP di azioni reali:
per questo micro-F1 (52,90%) differisce leggermente dall'accuracy (52,83%).

Non sono prodotte versioni L0/L1: manca una gerarchia semantica Ground definita.
Con una sola etichetta per frame-persona, subset accuracy e Jaccard per campione
coincidono con l'accuracy, trattando astensioni/lacune come insieme vuoto; una
matrice di co-occorrenza GT×predizione coincide con la matrice di confusione.
Non si applica NO_CARRY perché CarryingWhileLocomoting non è una classe Ground;
NO_IDLE e NO_IDLE_HOLD sono analisi aggiuntive, non equivalenti semantici.
Il testo TUMKitchen indica 25 Hz; Ground qui assume 30 Hz: i supporti grezzi
non sono confrontabili direttamente e Ground conta frame-persona, non frame
multilabel di una singola persona. Cambiano anche vocabolario e protocollo di
identità: le percentuali non costituiscono un confronto controllato fra dataset.

Eseguire `evaluation/Ground/evaluate_actions.py` passando il percorso dello ZIP
come primo argomento e `--output` seguito dalla cartella di destinazione.
Richiede soltanto Python standard. I risultati dettagliati sono nei CSV e
`summary.json`; i file originali non vengono modificati.
