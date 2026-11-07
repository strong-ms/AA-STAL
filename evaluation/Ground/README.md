# Ground: activity recognition da solo video

Branch di partenza: `eval/ground`, commit `85b04ae`. Questo documento definisce
il vocabolario candidato; non attesta prestazioni del modello o visibilità nei
video, che non sono stati esaminati in questa analisi.

## Evidenza dai log

I cinque file XLSM forniti contengono 247 segmenti nel primo foglio annotato.
`ground_truth_actions.csv` conserva file, foglio, riga, persona e valori originali.
La tassonomia e il foglio `Copy of Annotations` non sono segmenti aggiuntivi.
I nomi V1/V2 sono conservati senza assumere che rappresentino revisioni o viste
della stessa registrazione: chiarirlo prima di definire split di valutazione.

| Verbo GT → etichetta modello | Segmenti | Evidenza visiva richiesta |
|---|---:|---|
| Grasp → grasp | 51 | Passaggio da mano libera a presa sull'oggetto |
| Place_Down → place down | 44 | Appoggio su superficie seguito da rilascio |
| Hold → hold | 39 | Oggetto mantenuto in mano, senza manipolazione dominante |
| Idle → idle | 35 | Assenza visibile delle manipolazioni annotate |
| Twist → twist | 24 | Rotazione relativa di tappo/coperchio e contenitore |
| Drop → drop | 13 | Rilascio con caduta, distinto dall'appoggio controllato |
| Handover → handover | 12 | Trasferimento diretto fra persone, anche ricezione |
| Pour → pour | 12 | Inclinazione verso un recipiente e indizi di versamento |
| Cut → cut | 5 | Movimento di taglio del coltello sull'ingrediente |
| Shake → shake | 5 | Movimento ripetuto avanti/indietro del contenitore |
| Squash → squash | 4 | Pressione/schiacciamento con utensile nel recipiente |
| Push → push | 3 | Spostamento lungo una superficie senza sollevamento |

Sono tutte azioni candidate osservabili da una sequenza video, ma con difficoltà
diverse: prese, piccoli rilasci, torsioni e spinte richiedono dettaglio delle mani
e transizioni ben campionate. `Hold` e `Idle` sono stati, inclusi per coprire le
annotazioni. `Stir` è elencato in Action Taxonomy ma non compare nei 247 segmenti:
non è incluso nel vocabolario di questa valutazione.

Non inferire dal video la composizione di liquidi (`Alcohol`, `Cocktail`),
l'identità certa di ingredienti minuti o l'intento. I nomi oggetto provenienti
dalla detection sono indizi visivi, non etichette GT da fornire al modello.
Una finestra occlusa o ambigua deve produrre `unknown`, non automaticamente `idle`.

## Collaborazione e qualità delle annotazioni

`Collaborative_Drop`, `Collaborative_Handover`, `Collaborative_Pour` e
`Collaborative_Twist` sono relazioni fra partecipanti, da valutare separatamente
dal verbo della persona. Chi sostiene lo shaker durante un versamento resta
`Hold`, non diventa `Pour`. L'associazione geometrica persona-oggetto non prova
una collaborazione. Il riconoscimento delle relazioni richiederebbe osservare
entrambi i partecipanti e il medesimo oggetto nel tempo; non è attivato qui.

I fogli includono intestazioni di persone senza segmenti: non sono esempi Idle
e non vanno completate automaticamente. In `P000301_T1_V1`, Annotations riga 42,
`Collaborative_Drop` non specifica il partner. In `P000103_T1_V2`, riga 40,
`Handover` non ha una relazione collaborativa. Conservare questi valori senza
correggerli per supposizione. Alcuni `Squash` hanno Noun vuoto: il verbo rimane
valido senza inventare l'oggetto mancante.

## Configurazione pronta

Eseguire `data_pipeline/jupiter_notebooks/action_labeling.ipynb` dopo detection
e person-object association. Il notebook mantiene finestre da 60 frame, passo
30 e 20 frame campionati, con FPS sorgente impostato a 30: verificare che
corrisponda ai video effettivi. Il contesto del fotogramma intero è mantenuto.
Gli oggetti evidenziati sono quelli associati alla persona bersaglio.

Il prompt ora richiede un verbo dominante per finestra, senza la precedente
istruzione di scegliere due azioni per braccio. Le classi mantengono il contratto
`{"frames_start_to_end": {"actions": ["label"]}}`. Gli output e l'archivio
usano `ground_action_recognition_v1`, distinto dai risultati preesistenti.
Per rilanciare una configurazione diversa usare una nuova cartella di output:
il notebook salta i file per persona già presenti.

## Prima della valutazione quantitativa

1. Verificare unità e convenzione di Start/End nei log, FPS originali e mapping
   verso i frame estratti. I numeri sono preservati nel CSV senza conversioni.
2. Mappare esplicitamente Person A/B/C ai track ID per ciascun video; non assumere
   che l'ordine di detection corrisponda all'ordine dei partecipanti.
3. Convertire GT e predizioni nella stessa base temporale e scegliere una regola
   di confronto per le finestre che attraversano più segmenti. Il prompt sceglie
   il verbo temporalmente dominante; i confini evento richiedono verifica dedicata.
4. Riportare precision, recall, F1 e supporto per classe, macro-F1 e matrice di
   confusione, oltre al tasso di `unknown`. Le classi rare hanno 3–5 segmenti:
   i soli conteggi non permettono conclusioni robuste sulle prestazioni.

Questa preparazione non esegue Qwen e non produce metriche: l'inferenza richiede
l'ambiente GPU, i fotogrammi e le associazioni. Non usa le annotazioni GT come
input del riconoscimento.
