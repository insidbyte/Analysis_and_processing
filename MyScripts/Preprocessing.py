import re
import sys
import numpy as np
import pandas as pd
import os
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import spacy
import string
import plotly.offline as py
from Analyses_2 import Analyses_2
stop_words = stopwords.words('english')
py.init_notebook_mode(connected=True)

"""
IL PREPROCESSING CHE EFFETTUA QUESTO ALGORITMO E' DAVVERO BLANDO E SERVE SOLAMENTE A SOSTITUIRE LE FORME CONTRATTE DELLA 
LINGUA INGLESE E AD ELIMINARE PUNTEGGIATURA, CARATTERI SPECIALI, SINTASSI HTML, MAIL E SITI WEB.
QUESTO ALGORITMO SI PREOCCUPA MAGGIORMANTE DI ANALIZZARE IL DATASET E PUO MOSTRARE LE SEGUENTI ANALISI:
1)-NUMERO RECENSIONI POSITIVE E NEGATIVE
2)-NUMERO PAROLE NELLE RECENSIONI POSITIVE E NEGATIVE
3)-NUMERO DELLE STOPWORDS NELLE RECENSIONI POSITIVE E NEGATIVE
4)-LE PAROLE PIU' SIGNIFICATIVE PER WORD CLOUD
5)-LE N (SPECIFICATE NEL CODICE SORGENTE)PAROLE RIPETUTE PIU' VOLTE NEL DATASET
6)-GLI N GRAMMI (SPECIFICATI NEL CODICE SORGENTE) RIPETUTI PIU' VOLTE NEL DATABASE
L'ALGORITMO PUO' INOLTRE DIVIDERE IL DATASET IN TUTTE - SOLO POSITIVE E SOLO NEGATIVE PER ANALIZZARE O PROCESSARE
IN MANIERA DIFFERENTE O UGUALE LE RECENSIONI CON TARGET DIVERSO 
"""


class Analyses:
    def __init__(self, partialAnalyses, newProcessing, frac, dfPath, lemmatizza):
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    _________________________________________________________
                                    |                                                       |
                                    |TUTTE LE VARIABILI CREATE E MODIFICATE NEL COSTRUTTORE |
                                    |_______________________________________________________|
                
        *************************************************************
        *                                                           *
        *            VARIABILI LEGATE ALLA DIVISIONE DEL            *
        *               DATASET PER I PLOT DI ANALISI               *
        *                                                           *
        *************************************************************
        self.good_reviews -> dataframe review positive (sempre definito)
        self.bad_reviews -> dataframe review negative (sempre definito)
        self.count -> dataframe review totali -> (definito solo se newProcessing == False)
        self.count_good -> dataframe review buone -> (definito solo se option è 1 o 3)
        self.count_bad -> dataframe review cattive -> (definito solo se option è 1 o 2)
        self.count_good_words -> parole review buone -> (definito solo se newProcessing == False e option == 1 o 3)
        self.count_bad_words -> parole review buone -> (definito solo se newProcessing == False e option == 1 o 2)
        self.count_good_punctuations -> punteggiatura review buone -> (definito solo se newProcessing == False e 
                                                                        option == 1 o 3)
        self.count_bad_punctuations -> punteggiatura review buone -> (definito solo se newProcessing == False e 
                                                                option == 1 o 2)
        
        
        *************************************************************
        *                                                           *
        *               VARIABILE Dataframe pandas E                *
        *            ALTRE UTILI ALLA PULIZIA DEL TESTO             *
        *                                                           *
        *************************************************************
        self.df -> dataframe specificato da input -> (sempre definito)
        self.nlp -> spacy.load(disable, parser)->(sempre definito)
        self.stop -> stopList -> (sempre definito)
        self.sub -> re.compile(regex) -> (sempre definito)
        self.sub1 -> Se decidiamo di processare questo array viene popolato da tutti i campi di due parole
                     presenti a destra delle righe del file regex (self.setArrayRegex svolge questa funzione)
        self.present -> Se decidiamo di processare questo array viene popolato da tutti i campi di una parola
                     presenti a sinistra delle righe del file regex (self.setArrayRegex svolge questa funzione)
        self.path -> path del dataset -> (sempre definito)
        describe -> variabile dataframe.describe() utile per vedere se il dataset ha duplicati
        ********************************************************************************************
        *                                                                                          *
        *               VARIABILI IMPORTANTI PER LA DIVISIONE COME optione 1, 2, 3 E               *
        *                            ANALISI CON mask DI wordCloud                                 *
        *                                                                                          *
        ********************************************************************************************
        self.option -> input -> (sempre definito) -> Può essere 1 2 o 3 dipende se all negative o positive
        self.colorFrame -> sempre definito default white green se self.option == 3 red se self.option == 2
        self.frameCloud -> definito in base a self.option all positive o negative : 1 2 o 3
        
                                        ____________________________________________
                                        |                                          |
                                        |TUTTE LE VARIABILI PASSATE AL COSTRUTTORE |
                                        |__________________________________________|
                                        
        class Analyses:
            def __init__(self, partialAnalyses, newProcessing, frac, dfPath, lemmatizza):
            
        partialAnalyses -> BOOLEAN -> (Verificato solo se newProcessing == True accetta solo input 2 o 3 di self.option)
        newProcessing -> BOOLEAN -> (Sempre definito)
        frac -> FLOAT PER DEFINIRE LA PERCENTALE DA PROCESSARE
        dfPath -> STRING PER DAFINIRE LA PATH IN CUI E' SITUATO IL DATASET
        lemmatizza -> BOOLEAN (setta newPreprocessing == False nel main)
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        if partialAnalyses is True:
            print("Scegliere quale tipo di recensioni si vogliono considerare \n"
                  "1)-All (Sia negative che positive)\n"
                  "2)-Negative\n"
                  "3)-Positive")
            self.option = input()
        else:
            self.option = '1'
        if self.option != '1' and self.option != '2' and self.option != '3':
            sys.exit("Opzione non corretta SYSTEM EXIT !")
        # RANDOM SEED 24 PER AVERE SEMPRE CAMPIONI UGUALI
        np.random.seed(24)
        self.stop = self.getDefaultStoplist()
        self.sub = re.compile(r"<[^\S>]+>|[^A-Za-z@]|\S+@\S+|[^\w\s]|http+\S+|www+\S+")
        self.path = dfPath
        c = 0
        if lemmatizza is True:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            print(f"[{c}]-Lemmatizzazione avviata...")
            self.df = pd.read_csv(dfPath)
            self.df = self.df.sample(frac=1)
            print(f'\n[{c}]-START AT {datetime.now().strftime("%H:%M:%S")}')
            self.df['review'] = self.df['review'].apply(self.lemmatization)
            c = c+1
            print(f'\n[{c}]-FINISH AT {datetime.now().strftime("%H:%M:%S")}')
            print(f"[{c}]-Lemmatizzazione completata !")
            self.setDf(self.df)

        if newProcessing is True:
            print(f"[{c}]-Preparazione array di regex in corso...")
            self.present, self.sub1 = self.setArrayRegex()
            c = c+1
            print(f"[{c}]-Preparazione array di regex completata !")
            c = c+1
            print(f"[{c}]-Nuovo preprocessing avviato...")
            self.df = pd.read_csv("../Dataset/IMDB Dataset.csv")
            self.df = self.df.sample(frac=1)
            c = c+1
            print(f"[{c}]-Ricerca duplicati nel dataset...")
            describe = self.df.describe()
            if describe.iloc[1]['review'] != describe.iloc[0]['review']:
                c = c+1
                print(f"[{c}]-Duplicati trovati pulizia in corso...")
                self.df.drop_duplicates(inplace=True)
                c = c+1
                print(f"[{c}]-Pulizia terminata !")
            else:
                c = c+1
                print(f"[{c}]-Duplicati non trovati !")
            c = c+1
            print(f"[{c}]-Campionamento del {frac * 100}% in corso...")
            self.df = self.df.sample(frac=frac)
            c = c+1
            print(f"[{c}]-Campionamento del {frac * 100}% completato !")
            if partialAnalyses is True:
                if self.option == '2':
                    c = c+1
                    print(f'[{c}]-Selezione di soli elementi negativi in corso...')
                    self.count_bad = self.df[self.df['sentiment'] == 'negative']
                    self.df = self.df[self.df['sentiment'] == 'negative']
                    self.setDf(self.df)
                    c = c+1
                    print(f'[{c}]-Selezione di soli elementi negativi completata !')
                    c = c+1
                    print(f'[{c}]-Elementi solo negativi da analizzare: {len(self.count_bad)}')

                if self.option == '3':
                    c = c+1
                    print(f'[{c}]-Selezione di soli elementi positivi in corso...')
                    self.count_good = self.df[self.df['sentiment'] == 'positive']
                    self.df = self.df[self.df['sentiment'] == 'positive']
                    self.setDf(self.df)
                    c = c+1
                    print(f'[{c}]-Selezione di soli elementi positivi completata !')
                    c = c+1
                    print(f'[{c}]-Elementi solo positivi da analizzare: {len(self.count_good)}')

            if self.option == '1' and lemmatizza is False:
                c = c+1
                print(f'[{c}]-Selezione di elementi positivi e negativi in corso...')
                self.df = self.df.sample(frac=frac)
                self.count_good = self.df[self.df['sentiment'] == 'positive']
                self.count_bad = self.df[self.df['sentiment'] == 'negative']
                c = c+1
                print(f'[{c}]-Elementi da analizzare: {(len(self.count_good) + len(self.count_bad))}')

            c = c+1
            print(f'\n[{c}]-START PROCESSING OF {int(frac * 100)}% AT {datetime.now().strftime("%H:%M:%S")}')
            self.df['review'] = self.df['review'].apply(lambda z: self.removeWithRe(z))
            c = c+1
            print(f'\n[{c}]-FINISH PROCESSING AT {datetime.now().strftime("%H:%M:%S")}')
            self.setDf(self.df)

        elif lemmatizza is False:
            print("Si vogliono aggiungere le attuali words_stop ?\nY/N")
            decision = input().lower()
            self.df = pd.read_csv(dfPath)
            self.df = self.df.sample(frac=1)
            if decision == 'y':
                print("Si vuole bilanciare il dataset ? \nY/N")
                decision = input().lower()
                if decision == 'y':
                    c = c+1
                    print(f"[{c}]-Verifico che all' interno del dataset ci siano recensioni negative e positive...")
                    if self.option == '1':
                        c = c + 1
                        print(f"[{c}]-Recensioni negative e positive trovate !")
                        count_p = self.df[self.df['sentiment'] == 'positive']
                        count_n = self.df[self.df['sentiment'] == 'negative']
                        c = c + 1
                        if len(count_p) != len(count_n):
                            rus = RandomUnderSampler(random_state=24)
                            print(f"[{c}]-Dataset sbilanciato correzione in corso...")
                            self.df, self.df['sentiment'] = rus.fit_resample(self.df, self.df['sentiment'])
                            self.setDf(self.df)
                            count_p = self.df[self.df['sentiment'] == 'positive']
                            count_n = self.df[self.df['sentiment'] == 'negative']
                            print(f"[{c}]-Positive: {len(count_p)} - Negative: {len(count_n)}")
                            c = c+1
                            print(f"[{c}]-Dataset bilanciato !")
                        else:
                            print(f"[{c}]-Dataset bilanaciato azione non necessaria !")
                    else:
                        c = c+1
                        if self.option == '2':
                            target = 'negative'
                        elif self.option == '3':
                            target = 'positive'
                        print(f'[{c}]-Le recensioni sono solo {target} !')

                print(f'\n[{c}]-START AT {datetime.now().strftime("%H:%M:%S")}')
                self.aggiornaStop_words()
                print(f'\n[{c}]-FINISH AT {datetime.now().strftime("%H:%M:%S")}')

                print("Si vuole salvare il file ottenuto ?\nY/N")
                decision = input().lower()
                if decision == 'y':
                    file = pd.DataFrame(self.df)
                    times = str(datetime.now().strftime("%H_%M_%S_"))
                    if self.option == '1':
                        subdirectory = 'all/'
                    if self.option == '2':
                        subdirectory = 'negative/'
                    if self.option == '3':
                        subdirectory = 'positive/'
                    file.to_csv(f"../Dataset_processed/{subdirectory}/{times}no_stop.csv")
                    c = c+1
                    print(f"[{c}]-File: {times}no_stop.csv salvato in ../Dataset_processed/{subdirectory} !")

            if partialAnalyses is True:
                if self.option == '2':
                    c = c+1
                    print(f"[{c}]-Divisione di soli elementi negativi per l'analisi in corso...")
                    self.bad_reviews = self.df[self.df['sentiment'] == 'negative']['review']
                    self.good_reviews = None
                    self.count = self.df['sentiment'].value_counts()
                    self.count_good = None
                    self.count_bad = self.df[self.df['sentiment'] == 'negative']
                    self.count_good_words = None
                    self.count_bad_words = self.count_bad['review'].str.split().apply(self.cal_len)
                    self.count_good_punctuations = None
                    self.count_bad_punctuations = self.count_bad['review'].apply(lambda z: len([c for c in str(z)
                                                                                                if c in
                                                                                                string.punctuation]))

                    c = c+1
                    print(f"[{c}]-Divisione di soli elementi negativi per l'analisi completata !")
                if self.option == '3':
                    c = c + 1
                    print(f"[{c}]-Divisione di soli elementi positivi per l'analisi in corso...")
                    self.good_reviews = self.df[self.df['sentiment'] == 'positive']['review']
                    self.count = self.df['sentiment'].value_counts()
                    self.count_good = self.df[self.df['sentiment'] == 'positive']
                    self.count_bad = None
                    self.count_good_words = self.count_good['review'].str.split().apply(self.cal_len)
                    self.count_bad_words = None
                    self.count_good_punctuations = self.count_good['review'].apply(lambda z: len([c for c in str(z)
                                                                                                  if c in
                                                                                                  string.punctuation]))
                    self.count_bad_punctuations = None

                    c = c + 1
                    print(f"[{c}]-Divisione di soli elementi positivi per l'analisi completata !")
        if self.option == '1':
            c = c + 1
            print(f"[{c}]-Divisione di elementi negativi e positivi per l'analisi in corso...")
            self.good_reviews = self.df[self.df['sentiment'] == 'positive']['review']
            self.bad_reviews = self.df[self.df['sentiment'] == 'negative']['review']
            self.count = self.df['sentiment'].value_counts()
            self.count_good = self.df[self.df['sentiment'] == 'positive']
            self.count_bad = self.df[self.df['sentiment'] == 'negative']
            self.count_good_words = self.count_good['review'].str.split().apply(self.cal_len)
            self.count_bad_words = self.count_bad['review'].str.split().apply(self.cal_len)
            self.count_good_punctuations = self.count_good['review'].apply(lambda z: len([c for c in str(z)
                                                                                          if c in string.punctuation]))
            self.count_bad_punctuations = self.count_bad['review'].apply(lambda z: len([c for c in str(z)
                                                                                        if c in string.punctuation]))

            c = c + 1
            print(f"[{c}]-Divisione di elementi positivi e negativi per l'analisi completata !")
        self.colorFrame = 'white'
        if imgName is not None:
            c = c+1
            if self.option == '1':
                print(f"[{c}]-Selezione della maschera per word-cloud per review positive e negative in corso...")
                self.colorFrame = 'white'
                self.frameCloud = self.df['review']
            if self.option == '3':
                print(f"[{c}]-Selezione della maschera per word-cloud per review positive in corso...")
                self.frameCloud = self.count_good['review']
                self.colorFrame = 'green'
            if self.option == '2':
                print(f"[{c}]-Selezione della maschera per word-cloud per review negative in corso...")
                self.frameCloud = self.count_bad['review']
                self.colorFrame = 'red'
            c = c+1
            print(f"[{c}]-Selezione della maschera per word-cloud completata !")

    def setDf(self, df):
        """
        Aggiorna il dataframe con il valore passato
        """
        self.df = df

    def cal_len(self, data):
        """
        ritorna la lunghezza del dato passato
        """
        if isinstance(data, float):
            return 0
        return len(data)

    # Stampa grandezza dataframe
    def printShapeOfDf(self):
        """
        stampa quante review negative e positive ha attualmente il dataset
        """
        print("The Shape of the Dataset".format(), self.df.shape)


    def plotCount(self):
        """
        mostra con plot le review negatve e positive presenti nel dataset
        """
        print('Total Counts of both sets'.format(), self.count)
        print("==============")
        plt.rcParams['figure.figsize'] = (6, 6)
        if self.count_good is not None:
            plt.bar(0, len(self.count_good), width=0.6, label='Positive Reviews', color='Green')
        plt.legend()
        if self.count_bad is not None:
            plt.bar(2, len(self.count_bad), width=0.6, label='Negative Reviews', color='Red')
        plt.legend()
        plt.ylabel('Count of Reviews')
        plt.xlabel('Types of Reviews')
        plt.show()

    def plotCountWords(self):
        """
        mostra con plot le parole presenti nel dataset
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        if self.count_bad_words is not None:
            sns.histplot(self.count_bad_words, ax=ax1, color='Red')
            ax1.set_title("Negative Review")
        if self.count_good_words is not None:
            sns.histplot(self.count_good_words, ax=ax2, color='Green')
            ax2.set_title("Positive Review")
        fig.suptitle("Reviews Word Analysis")
        plt.show()

    def plotCountPunct(self):
        """
        mostra con plot la punteggiatura attualmente presente nel dataset
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        if self.count_bad_punctuations is not None:
            sns.histplot(self.count_bad_punctuations, ax=ax1, color='Red')
            ax1.set_title("Negative Review Punctuations")
        if self.count_good_punctuations is not None:
            sns.histplot(self.count_good_punctuations, ax=ax2, color='Green')
            ax2.set_title("Positive Review Punctuations")
        fig.suptitle("Reviews Word Punctuation Analysis")
        plt.show()


    def display_cloud(self, imgPath):
        """
        Mostrerà le parole più significative con o senza mask per word-cloud
        se imgPath è none vuol dire che nel main abbiamo deciso di non utilizzare una maschera per word-cloud
        """
        plt.subplots(figsize=(10, 10))
        if imgPath is not None:
            mask = np.array(Image.open(imgPath))
        else:
            mask = None
        if mask is not None:
            wc = WordCloud(stopwords=STOPWORDS,
                           mask=mask, background_color="black", contour_width=2, contour_color=self.colorFrame,
                           max_words=2000, max_font_size=256,
                           random_state=42, width=mask.shape[1],
                           height=mask.shape[0])
            wc.generate(' '.join(self.frameCloud))
        else:
            wc = WordCloud(stopwords=STOPWORDS,
                           background_color="black", contour_width=2, contour_color=self.colorFrame,
                           max_words=2000, max_font_size=256,
                           random_state=42)
            wc.generate(' '.join(map(str, self.df['review'])))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def commonWords(self, ngrams, array, n):
        """
        Se ngrams è False prenderà le [:n] parole singole più comuni del dataset
        altrimenti prenderà le frasi di lunghezza specificata nel main più comuni del dataset
        """
        if ngrams is False and array is None:
            if self.option == '1':
                corpus = [i for x in analises.good_reviews.str.split() for i in x]
                corpus.append(i for x in analises.bad_reviews.str.split() for i in x)
            if self.option == '2':
                corpus = [i for x in analises.bad_reviews.str.split() for i in x]
            if self.option == '3':
                corpus = [i for x in analises.good_reviews.str.split() for i in x]
        else:
            corpus = array
        counter = Counter(corpus)
        most = counter.most_common()
        x = []
        y = []
        if array is None:
            for word, count in most[:n]:
                if word not in self.stop:
                    x.append(word)
                    y.append(count)
        else:
            for word, count in most[:]:
                if word not in self.stop:
                    x.append(word)
                    y.append(count)
        sns.barplot(x=y, y=x)
        plt.show()

    def getDefaultStoplist(self):
        """
        Questa funzione viene utilizzata all'inizio del costruttore e setta le stop_words al completo viene
        utilizzata quando :
        Lemmatizzo e scelgo di aggiungere le stop_words personalizzare
        Se invece analizzo il dataset e scelgo di aggiungere le stop_words non userò queste liste ma quelle presenti in
        self.aggiornaStop_Words
        """
        stopwords = nltk.corpus.stopwords.words('english')
        with open('../Stop_words/stopSecondWords.txt', 'r') as f:
            x_gl = f.readlines()
        with open('../Stop_words/weekWords.txt', 'r') as f:
            x_n = f.readlines()
        with open('../Stop_words/months.txt', 'r') as f:
            x_d = f.readlines()
        with open('../Stop_words/name.txt', 'r') as f:
            x_no = f.readlines()
        with open('../Stop_words/stopWords.txt', 'r') as f:
            x_1 = f.readlines()
        with open('../Stop_words/stopwords_google.txt', 'r') as f:
            x_2 = f.readlines()
        with open('../Stop_words/html.txt', 'r') as f:
            x_3 = f.readlines()
        with open("../Stop_words/commonInDataset.txt", 'r') as f:
            x_4 = f.readlines()
        if self.option == '2':
            with open("../Stop_words/negative/positive_words.txt") as f:
                x_5 = f.readlines()
                [stopwords.append(x.rstrip()) for x in x_5]
        if self.option == '3':
            with open("../Stop_words/positive/negative_words.txt") as f:
                x_5 = f.readlines()
                [stopwords.append(x.rstrip()) for x in x_5]
        [stopwords.append(x.rstrip()) for x in x_gl]
        [stopwords.append(x.rstrip()) for x in x_n]
        [stopwords.append(x.rstrip()) for x in x_d]
        [stopwords.append(x.rstrip()) for x in x_no]
        [stopwords.append(x.rstrip()) for x in x_1]
        [stopwords.append(x.rstrip()) for x in x_2]
        [stopwords.append(x.rstrip()) for x in x_3]
        [stopwords.append(x.rstrip()) for x in x_4]
        return set([word.lower() for word in stopwords])



    def setArrayRegex(self):
        """
        Viene utilizzata solo se scelgo di processare nel main
        definisce e ritorna due array generati dal file regex
        che dovrà obbligatoriamente contenere una di queste due sitassi:
        1)-espressione_da_cercare – espressione_da_sostituire (se quella da sostituire è composta da due parole)
        2)-espressione_da_cercare – espressione_da_sostituire , (se quella da sostituire è composta da una parola)
        """
        present = []
        sub = []
        with open("../regex", 'r') as f:
            l = f.readlines()
            list = [re.sub(r"[â€™]", "'", s) for s in l]
            list = [re.sub(r"A-za-z|[â€]|[\n]", "", s) for s in l]
        list = [re.sub(r"™", "'", s) for s in list]
        for l in list:
            contract = l.split(' ')
            append_all = False
            if len(l) != 0:
                present.append(contract[0])
                for s in string.punctuation:
                    if contract[3] == s:
                        append_all = False
                        break
                    else:
                        append_all = True
                if append_all is False:
                    sub.append(contract[2])
                elif append_all is True:
                    sub.append(contract[2] + " " + contract[3])
        return present, sub

    def removeWithRe(self, text):
        """
        Viene utilizzata solo se scelgo di processare nel main
        self.present e self.sub1 sono due array che contengono lo stesso numero di elementi
        se viene trovata una espressione presente in self.present[i] o le sue alterative verrà sostituita con
        l'espressione presente in self.sub1[i] tramite sub del modulo re
        self.sub invece è un re.compile() che elimina linguaggio html, punteggiatura, siti, mail e altro
        """
        words = text.split(" ")
        words = [word for word in words if word != ' ' or word != '  ' or word !='    ' or word != '']
        i = 0
        while i < len(self.present):
            alternative = re.sub("'", "’", self.present[i])
            alternative_2 = re.sub("'", "´", self.present[i])
            alternative_3 = re.sub("'", "`", self.present[i])
            alternative_4 = re.sub("'", "", self.present[i])
            alternative_5 = re.sub("'", ",", self.present[i])
            alternative_6 = re.sub("'", ";", self.present[i])
            alternative_7 = re.sub("'", ":", self.present[i])
            alternative_8 = re.sub(" ", "", self.sub1[i])
            array_patterns = [self.present[i], alternative, alternative_2, alternative_3, alternative_4, alternative_5,
                              alternative_6, alternative_7, alternative_8]
            combinated_patterns = r'|'.join(map(r'(?:{})'.format, array_patterns))

            words = [re.sub(combinated_patterns, self.sub1[i], word.lower()) for word in words]
            i = i+1

        words = [re.sub(self.sub, " ", word) for word in words]
        string = " ".join(word.strip() for word in words if word.strip())
        return string

    def update_Stop(self, text):
        """
        Questo metodo viene chiamato quando decidiamo di analizzare un file e di aggiungerci le stop_words presenti in
        self.aggiornaStop_words
        """
        words = text.split(" ")
        words = [word.lower() for word in words if word.lower() not in self.stop]
        words = self.removeRepeat(words)
        string = " ".join(words)
        return string
    #64,33
    def aggiornaStop_words(self):
        """
        In questa funzione è possibile specificare delle wordlist per escludere le parole nel corpus
        nel momento in cui decidiamo di analizzare un dataset già processato e decidiamo di usare le
        stopwords aggiornate
        """
        stopwords = []
        with open('../Stop_words/weekWords.txt', 'r') as f:
            x_n = f.readlines()
        with open('../Stop_words/html.txt', 'r') as f:
            x_h = f.readlines()
        with open('../Stop_words/months.txt', 'r') as f:
            x_d = f.readlines()
        with open('../Stop_words/name.txt', 'r') as f:
            x_no = f.readlines()
        with open("../Stop_words/inutili.txt", 'r') as f:
            x_4 = f.readlines()
        with open("../Stop_words/commonInDataset.txt") as f:
            x_5 = f.readlines()

        [stopwords.append(x.rstrip()) for x in x_n]
        [stopwords.append(x.rstrip()) for x in x_d]
        [stopwords.append(x.rstrip()) for x in x_no]
        [stopwords.append(x.rstrip()) for x in x_4]
        [stopwords.append(x.rstrip()) for x in x_5]
        [stopwords.append(x.rstrip()) for x in x_h]

        if self.option == '2':
            with open("../Stop_words/negative/positive_words.txt", 'r') as f:
                x_5 = f.readlines()
                [stopwords.append(x.rstrip()) for x in x_5]
        if self.option == '3':
            with open("../Stop_words/positive/negative_words.txt", 'r') as f:
                x_5 = f.readlines()
                [stopwords.append(x.rstrip()) for x in x_5]

        self.stop = set([word.lower() for word in stopwords])
        #self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        #self.df = pd.read_csv(self.path)
        print(f'\nSTART PROCESSING AT {datetime.now().strftime("%H:%M:%S")}')
        self.df['review'] = self.df['review'].apply(self.update_Stop)
        print(f'\nFINISH PROCESSING AT {datetime.now().strftime("%H:%M:%S")}')
        self.setDf(self.df)

    def lemmatization(self, data):
        """
        Questa funzione viene chiamata quando decidiamo di lemmatizzare nel main e utilizzerà il modulo spacy
        en_core_web_sm
        """
        words = [token.lemma_ for token in self.nlp(data) if token.text != '' or '  ']
        #words = set([str(token) for token in model if not token.is_punct if token.text != ' ' if token.text != ''])
        string = " ".join(words)
        return string

    def removeRepeat(self, words):
        c = 0
        new_words = []
        words = [word for word in words if word != '' and word != ' ']
        # 64,34
        prev_word = ''
        for word in words:
            if c == 0:
                first = True
            else:
                first = False
            if c != len(words) - 1:
                b = c + 1
                if first is False:
                    a = c - 1
                c = c + 1
            else:
                a = c - 1
                prev_word = words[a]
                if word != prev_word and word != prev_word + " " and word != " " + prev_word:
                    new_words.append(word)
                break
            next_word = words[b]
            if first == False:
                prev_word = words[a]
            if first is False and word != next_word \
                              and word != prev_word \
                              and word != next_word + " " \
                              and word != prev_word + " "\
                              and word != " "+next_word \
                              and word != " "+prev_word\
                              and c > 2:
                new_words.append(prev_word)
            if first is False and word == next_word and word != prev_word:
                new_words.append(prev_word)

            if first is True and word != next_word \
                             and word != next_word + " " \
                             and word != " " + next_word :
                new_words.append(word)
        return new_words

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
if __name__ == '__main__':
    print("Inserire un opzione:\n"
          "1)-UNIRE DUE DATASET\n"
          "2)-FARE UNA PRIMA PULIZIA DEL DATASET\n"
          "3)-ANALIZZARE DATASET O ELIMINARE STOP WORDS\n"
          "4)-LEMMATIZZARE DATASET")
    option = input()
    if option != '1' and option != '2' and option != '3' and option != '4':
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
        file = pd.DataFrame(df)
        print("Inserire nuovo nome del file unito di output (.csv escluso)")
        nome = input()
        path = "../Dataset_processed/all/"+nome+".csv"
        file.to_csv(path, index=False)
        nome = ''
        path = ''
        sys.exit(100)
    if option == '2':
        print("Digitare percentuale da processare come numero intero")
        frac = float(input()) / 100
        dfPath = "Dataset/IMDB Dataset.csv"
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

        path = "../Dataset_processed/all/"+nome+".csv"
        path1 = "../Dataset_processed/negative/"+nome+".csv"
        path2 = "../Dataset_processed/positive/"+nome+".csv"

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

    analises = Analyses(partialAnalyses=partialAnalyses, newProcessing=newPreprocessing, frac=frac, dfPath=dfPath,
                        lemmatizza=lemmatizza)

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
    print("Si vuole analizzare con i grafici il dataset lavorato ? \nY/N")
    decision = input().lower()
    if decision != 'y' and decision != 'n':
        sys.exit("Opzione errata SYSTEM EXIT !")
    if decision == 'y':
        analizza = True
    elif decision == 'n':
        analizza = False
    if newPreprocessing is False and analizza is True:
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
            analises.an_2 = Analyses_2(ngrams=ngrams, df=data)
            analises.an_2.createDict(data=analises.df)
            trace = analises.an_2.create_new_df(n=n)
            count = trace.x
            words = trace.y
            count = count[:70]
            words = words[:70]
            print(f'TRACE: {trace}')

            sns.barplot(x=count, y=words)
            plt.show()

