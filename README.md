# Analysis_and_processing
 
 __Esegue un analisi statistica del dataset e ha più opzioni:__ 
     
    1)-Unisce due dataset.
    2)-Fa una prima pulizia del dataset.
    3)-Analizza ed eventualmente elimina stop words.
    4)-Lemmatizza

## Le fasi vanno esguite con un ordine altrimenti il dataset di output non sarà attendibile !
## Prima fase:
#### Toglieremo caratteri speciali, siti web, e-mail, codice html e tutte le contratture della lingua inglese.
#### Per prima cosa andiamo nel file first e scriviamo alla prima riga : True.
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase_file_first.png)
#### Successivamente lanciamo Processing.py e inseriamo il seguente input.
#### Input:
![Screenshot](MyScripts/OUTPUTS/Fasi_di_pulizia/1a_Fase.png)
#### Output:










### Terza Fase:
#### Questa è la fase più importante perchè consente di alleggerire molto il dataset lemmatizzato e pulito.
#### Possiamo vedere quante review positive e negative ha il dataset ed eseguire word-cloud o una analisi ngrams.
#### Sotto vengono riportate alcune immagini che mostrano l'efficacia delle fasi precedenti e alcune informazioni 
#### preziose per costruire wordlist personalizzate.
## Conteggio review positive e negative:
![Screenshot](MyScripts/OUTPUTS/count_negative_positive.png)
