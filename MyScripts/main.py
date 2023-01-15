import multiprocessing
import sys
import os
import pandas as pd
import re
import numpy as np
from Analyses_2 import Analyses_2
from Preprocessing import Analyses
"""
Passaggi consigliati:

1)- Fare una prima pulizia:
    OPZIONE (2) da terminale questo pulirà il dataset sostituendo le forme contratte inglesi in forme estese
    e togliendo catteri speciali html, punteggiatura, spazi, siti web, e-mail e altre cose che non ci servono 
    per l'addestramento

2)- Lemmatizzare:
    OPZIONE (4) da terminale tutte le parole verrannò sostituite nella parola radice ad esempio, se è presente warning 
    diventerà warn e così fino alla fine questo ci aiuterà per rimuovere le stop_words è consigliabile non rimuovere le 
    stop_words quando il programma ce lo chiederà perchè sono troppe e eliminano troppi dati
    è consigliabile rimuovere le stop_words con l'analisi come descritto a breve

3)- Rimozione stop_words:
    OPZIONE (3) da terminale sarà possibile rimuovere le stop_words e salvare il file 

4)- Analisi e perfezionamento:
    OPZIONE(3) da terminale potremo dinamicamente valutare il dataset con diverse stop_words e vedere quali sono le più 
    efficaci (ad esempio tag html) decidendo se salvare o no il risultato

N.B.
OPZIONE(1) VIENE UTILIZZATA SOLO PER UNIRE DUE DATASET CHE CONTENGONO SOLO RECENSIONI POSITIVE O NEGATIVE
PUO' ESSERE UTILE SE DECIDIAMO DI APPLICARE DIVERSE STOP_WORDS PER I DUE SENTIMENT DIVERSI  

N.B
SE SI LANCIA IL PROGRAMMA LA PRIMA VOLTA BISOGNA SCRIVERE True in ../first
"""




path = None
imgName = None
lemmatizza = False
newPreprocessing = False
partialAnalyses = False
dfPath = ''
analizza = False
correggi = False
manda_dataset = False
if __name__ == '__main__':

    print(f"Number of CPU : {multiprocessing.cpu_count()}")
    print("Inserire un opzione:\n"
          "1)-UNIRE DUE DATASET\n"
          "2)-FARE UNA PRIMA PULIZIA DEL DATASET\n"
          "3)-ANALIZZARE DATASET O ELIMINARE STOP WORDS\n"
          "4)-LEMMATIZZARE DATASET\n"
          "5)-CORREGGERE IL DATASET\n"
          "6)-MANDARE FILE AL MODEL ADMIN")
    option = input()
    if option != '1' and option != '2' and option != '3' and option != '4' and option != '5' and option != '6':
        sys.exit("Opzione non corretta SYSTEM EXIT !")
    if option == '1':
        list = os.listdir("../Dataset_processed/negative")
        print("Inserire nome del dataset negativo da unire (.csv escluso)")
        for l in list:
            print(re.sub("\.csv", "", l))
        dfPath = "../Dataset_processed/negative/" + input() + ".csv"
        df1 = pd.read_csv(dfPath)
        print("Inserire nome del dataset positivo da unire (.csv escluso)")
        list = os.listdir("../Dataset_processed/positive")
        for l in list:
            print(re.sub("\.csv", "", l))
        dfPath = "../Dataset_processed/positive/" + input() + ".csv"
        df2 = pd.read_csv(dfPath)
        df = pd.concat([df1, df2], axis=0)
        df = df.sample(frac=1)
        file = pd.DataFrame(df)
        print("Inserire nuovo nome del file unito di output (.csv escluso)")
        nome = input()
        path = "../Dataset_processed/all/" + nome + ".csv"
        file.to_csv(path, index=False)
        nome = ''
        path = ''
        sys.exit(100)
    if option == '2':
        print("Digitare percentuale da processare come numero intero")
        frac = float(input()) / 100

        decision = input("Si vuole pulire un dataset già processato ?\n Y/N?\n").lower()
        if decision == 'y':
            list = os.listdir("../Dataset_processed/all")
            list += os.listdir("../Dataset_processed/negative")
            list += os.listdir("../Dataset_processed/positive")
            print("Inserire nome del file da analazzare (.csv escluso)")

            for l in list:
                print(re.sub("\.csv", "", l))

            nome = input()

            path = "../Dataset_processed/all/" + nome + ".csv"
            path1 = "../Dataset_processed/negative/" + nome + ".csv"
            path2 = "../Dataset_processed/positive/" + nome + ".csv"

            cond = False
            if os.path.exists(path):
                dfPath = path
                cond = True
            if os.path.exists(path1):
                dfPath = path1
                cond = True
            if os.path.exists(path2):
                dfPath = path2
                cond = True

            if cond is False:
                sys.exit("File non trovato SYSTEM EXIT !")
        else:
            dfPath = "../Dataset/IMDB Dataset.csv"

        newPreprocessing = True
        print("Si desidera processare parzialmente (solo Negative o Positive Review) ? \nY/N")
        decision = input().lower()
        if decision != 'y' and decision != 'n':
            sys.exit("Opzione errata SYSTEM EXIT !")
        if decision == 'y':
            partialAnalyses = True
        print("inserire nome del nuovo dataset da processare in output (.csv escluso)")
        nome = input()
    if option == '3':
        print("Si vuole utilizzare una maschera per wordCloud? \nY/N")
        decision = input().lower()
        if decision != 'y' and decision != 'n':
            sys.exit("Opzione errata SYSTEM EXIT !")
        if decision == 'y':
            print('\nSelezionare una path valida: ')
            for image in os.listdir("../input"):
                image = re.sub(r"\.png", "", image)
                print(image)
            imgName = input()
            pathMask = '../input/' + imgName + ".png"
        else:
            pathMask = None
        frac = 1

        list = os.listdir("../Dataset_processed/all")
        list += os.listdir("../Dataset_processed/negative")
        list += os.listdir("../Dataset_processed/positive")

        print("Inserire nome del file da analazzare (.csv escluso)")

        for l in list:
            print(re.sub("\.csv", "", l))

        nome = input()

        path = "../Dataset_processed/all/" + nome + ".csv"
        path1 = "../Dataset_processed/negative/" + nome + ".csv"
        path2 = "../Dataset_processed/positive/" + nome + ".csv"

        cond = False
        if os.path.exists(path):
            dfPath = path
            cond = True
        if os.path.exists(path1):
            dfPath = path1
            cond = True
        if os.path.exists(path2):
            dfPath = path2
            cond = True

        if cond is False:
            sys.exit("File non trovato SYSTEM EXIT !")

        analizza = True
        partialAnalyses = True

    if option == '4':
        print("Inserire nome del file da lemmatizzare (.csv escluso)")

        list = os.listdir("../Dataset_processed/all")
        list += os.listdir("../Dataset_processed/negative")
        list += os.listdir("../Dataset_processed/positive")

        for l in list:
            print(re.sub("\.csv", "", l))

        nome = input()

        print("Il file contiene recensioni solo positive o negative?\nY/N")
        decision = input().lower()
        if decision != 'y' and decision != 'n':
            sys.exit("Opzione non corretta SYSTEM EXIT !")
        if decision == 'y':
            partialAnalyses = True

        path = "../Dataset_processed/all/" + nome + ".csv"
        path1 = "../Dataset_processed/negative/" + nome + ".csv"
        path2 = "../Dataset_processed/positive/" + nome + ".csv"

        cond = False
        if os.path.exists(path):
            dfPath = path
            cond = True
        if os.path.exists(path1):
            dfPath = path1
            cond = True
        if os.path.exists(path2):
            dfPath = path2
            cond = True

        if cond is False:
            sys.exit("File non trovato SYSTEM EXIT !")

        print("inserire nome del nuovo dataset lemmatizato di output (.csv escluso)")
        nome = input()
        lemmatizza = True
        newPreprocessing = False
        frac = 1
    if option == '5':
        print("Inserire nome del file da correggere (.csv escluso)")

        list = os.listdir("../Dataset_processed/all")
        list += os.listdir("../Dataset_processed/negative")
        list += os.listdir("../Dataset_processed/positive")

        for l in list:
            print(re.sub("\.csv", "", l))

        nome = input()

        print("Il file contiene recensioni solo positive o negative?\nY/N")
        decision = input().lower()
        if decision != 'y' and decision != 'n':
            sys.exit("Opzione non corretta SYSTEM EXIT !")
        if decision == 'y':
            partialAnalyses = True

        path = "../Dataset_processed/all/" + nome + ".csv"
        path1 = "../Dataset_processed/negative/" + nome + ".csv"
        path2 = "../Dataset_processed/positive/" + nome + ".csv"

        cond = False
        if os.path.exists(path):
            dfPath = path
            cond = True
        if os.path.exists(path1):
            dfPath = path1
            cond = True
        if os.path.exists(path2):
            dfPath = path2
            cond = True

        if cond is False:
            sys.exit("File non trovato SYSTEM EXIT !")

        print("inserire nome del nuovo dataset corretto di output (.csv escluso)")
        nome = input()
        correggi = True
        newPreprocessing = False
        frac = 1

    if option == '6':
        print("Selezionare file csv da mandare (.csv escluso) :")

        list = os.listdir("../Dataset_processed/all")
        list += os.listdir("../Dataset_processed/negative")
        list += os.listdir("../Dataset_processed/positive")

        for l in list:
            print(re.sub("\.csv", "", l))

        nome = input()

        path = "../Dataset_processed/all/" + nome + ".csv"
        path1 = "../Dataset_processed/negative/" + nome + ".csv"
        path2 = "../Dataset_processed/positive/" + nome + ".csv"

        cond = False
        if os.path.exists(path):
            dfPath = path
            cond = True
        if os.path.exists(path1):
            dfPath = path1
            cond = True
        if os.path.exists(path2):
            dfPath = path2
            cond = True
        file = pd.read_csv(dfPath)
        frac = 1
        nome = input("Inserire nome del file di output da salvare in modelAdmin/Dataset_processed (.csv escluso):\n")
        file.to_csv(f"../../modelAdmin/Dataset_processed/{nome}.csv")
        sys.exit(200)

    analises = Analyses(partialAnalyses=partialAnalyses, newProcessing=newPreprocessing, frac=frac, dfPath=dfPath,
                        lemmatizza=lemmatizza, correggi=correggi, imgName=imgName)

    if nome != '' and analizza is False:
        subdirectory = ''
        if analises.option == '1':
            subdirectory = 'all/'
        if analises.option == '2':
            subdirectory = 'negative/'
        if analises.option == '3':
            subdirectory = 'positive/'
        file = pd.DataFrame(analises.df)
        print(f'Index: {file.index}')
        with open("../first", "r") as f:
            l = f.readlines()
            if l[0] == 'True':
                file.to_csv(f'../Dataset_processed/{subdirectory}/{nome}.csv')
                with open('../first', 'w') as f:
                    f.write('False')
            else:
                file.to_csv(f'../Dataset_processed/{subdirectory}/{nome}.csv', index=False)
        print(f"File: {nome}.csv salvato in Dataset_preprocessed/{subdirectory} !!")

    if option == '3':
        print("Si vuole analizzare con i grafici il dataset lavorato ? \nY/N")
        decision = input().lower()
        if decision != 'y' and decision != 'n':
            sys.exit("Opzione errata SYSTEM EXIT !")
        if decision == 'y':
            analizza = True
        elif decision == 'n':
            analizza = False

    if analizza is True and option == '3':
        print("Si vuole vedere quante review positive e negative ha il dataframe? \nY/N")
        decision = input().lower()
        if decision == 'y':
            analises.plotCount()
        print("Si vogliono vedere le parole positive e negative nel dataframe? \nY/N")
        decision = input().lower()
        if decision == 'y':
            analises.plotCountWords()
        print("Si vuole vedere quanta punteggiatura è presente nel dataframe? \nY/N")
        decision = input().lower()
        if decision == 'y':
            analises.plotCountPunct()

        print("Si vuole eseguire word-cloud ? \nY/N")
        decision = input().lower()
        if decision == 'y':
            analises.display_cloud(imgPath=pathMask)
        if analises.stop is not None:
            print("Si vogliono vedere le parole positive e negative più comuni nel dataframe? \nY/N")
            decision = input().lower()
            if decision == 'y':
                print("Specificare quante mostrarne come numero intero: ")
                n = int(input())
                analises.commonWords(ngrams=False, array=None, n=n)
        print("Si vuole eseguire ngram ? \nY/N")
        decision = input().lower()
        if decision == 'y':
            """
            E' possibile specificare i ngrams in ngrams=numero
            E' possibile dire quante frasi più comuni ricercare specificando l'int della quantità
            in count[:numero] words[:numero] questo numero deve essere uguale al numero specificato nella funzione
            della classe Analyses_2 : create_new_df() quando passiamo il parametro freq_df[:numero]
            """
            print("Specificare quante parole per frase si vogliono considerare come numero intero: ")
            ngrams = int(input())
            print("Specificare quante mostrarne come numero intero: ")
            n = int(input())
            data = analises.df['review']
            ngrams = 1
            items = []
            items2 = []
            items3 = []
            while ngrams <= 3:
                analises.an_2 = Analyses_2(ngrams=ngrams, df=data)
                analises.an_2.createDict(data=analises.df)
                if ngrams == 1:
                    trace = analises.an_2.create_new_df(n=int(n / 2))
                else:
                    trace = analises.an_2.create_new_df(n=n)
                count = trace.x
                words = trace.y
                count = count[:n]
                words = words[:n]
                if ngrams == 1:
                    for word in words:
                        items.append(word)
                if ngrams == 2:
                    for word in words:
                        items2.append(word)
                if ngrams == 3:
                    for word in words:
                        items3.append(word)
                with np.printoptions(threshold=np.inf):
                    print(f'TRACE: {trace.x}')
                with np.printoptions(threshold=np.inf):
                    print(f'TRACE: {trace.y}')
                # sns.barplot(x=count, y=words)
                # plt.show()
                ngrams = ngrams + 1

            analises.dumpVoc(items, items2, items3)
