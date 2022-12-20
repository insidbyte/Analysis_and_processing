# Analysis_and_processing
 
 __Esegue un analisi statistica del dataset e ha più opzioni:__ 
     
    1)-Unisce due dataset.
    2)-Fa una prima pulizia del dataset.
    3)-Analizza ed eventualmente elimina stop words.
    4)-Lemmatizza

## Le fasi vanno esguite con in ordine altrimenti il dataset di output non sarà attendibile !
## Prima fase:
#### Toglieremo caratteri speciali, siti web, e-mail, codice html e tutte le contratture della lingua inglese.
#### Per prima cosa andiamo nel file first e scriviamo alla prima riga : True.
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_file_first.png)
#### Se desideriamo aggiungere espressioni da sostituire tramite regex dobbiamo aprire il file regex e modificarne il contenuto.
#### La sintassi coretta per la sostituizione è:
#### espressione_da sostituire – espressione_da_sostituire_con_lo_spazio
#### oppure se l'espressione da sostituire ha una sola parola
#### espressione_da sostituire – espressione_da_sostituire ,
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/util.png)
#### Successivamente lanciamo Processing.py.
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_lunch.png)
#### Poi inseriamo il seguente Input:
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase.png)
#### Output:
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_output1.png)
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_outputb.png)
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_output2.png)
#### Input:
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_input.png)

## Seconda fase:
#### Lemmatizziamo sfoltendo un po il datset e sostituendo ogni parola composta con la propria radice
#### Input:
![Screenshot](MyScripts/OUTPUTS/lemmatizzazione/2a_Fase_input.png)
#### Output
![Screenshot](MyScripts/OUTPUTS/lemmatizzazione/2a_Fase_output1.png)
![Screenshot](MyScripts/OUTPUTS/lemmatizzazione/2a_Fase_output2.png)
#### Input:
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_input.png)

## Terza Fase:
#### In questo caso non è neccesaria ma nel caso in cui avessimo pulito e lemmatizzato solo le review positive o 
#### negative, dobbiamo unire il dataset per procedere alla fase di analisi.

## Quarta Fase:
#### Questa è la fase più importante perchè consente di alleggerire molto il dataset lemmatizzato e pulito.
#### Per aggiungere nuove stopwords oltre quelle già presenti nel repository basta aggiungere le parole nei file di testo:
![Screenshot](MyScripts/OUTPUTS/stopwords/stopwords.png)
#### Input:
![Screenshot](MyScripts/OUTPUTS/4a_fase/4a_Fase_input.png)
#### Output
![Screenshot](MyScripts/OUTPUTS/4a_fase/4a_Fase_output.png)
![Screenshot](MyScripts/OUTPUTS/4a_fase/4a_Fase_output1.png)
#### Possiamo vedere quante review positive e negative ha il dataset ed eseguire word-cloud o una analisi ngrams.
#### Sotto vengono riportate alcune immagini che mostrano l'efficacia delle fasi precedenti e alcune informazioni 
#### preziose per costruire wordlist personalizzate.

### Conteggio review positive e negative:
![Screenshot](MyScripts/OUTPUTS/count_negative_positive.png)

### Parole più significative per Word Cloud:
### Negative
![Screenshot](MyScripts/OUTPUTS/word_cloud_negative.png)
### Positive
![Screenshot](MyScripts/OUTPUTS/word_cloud_positive.png)

### Parole più comuni nel dataset:
### Positive
![Screenshot](MyScripts/OUTPUTS/most_common50_positive.png)
### Negative
![Screenshot](MyScripts/OUTPUTS/most_common_negative.png)

### Parole più comuni nel dataset con NGRAMS 2:
![Screenshot](MyScripts/OUTPUTS/ngrams2_negative_top50.png)

# CONCLUSIONI:
