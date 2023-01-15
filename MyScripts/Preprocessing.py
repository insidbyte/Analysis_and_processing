import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import spacy
import string
from joblib import dump
from textblob import TextBlob
from multiprocessing import Process, Manager, Lock
from Multiprocessing import Multiprocessing

stop_words = stopwords.words('english')
thread_cont = 0
thread_name = ''
c = 0

"""
LA PRIMA PULIZIA CHE EFFETTUA QUESTO ALGORITMO E' DAVVERO BLANDA E SERVE SOLAMENTE A SOSTITUIRE LE FORME CONTRATTE DELLA 
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
    def __init__(self, partialAnalyses, newProcessing, frac, dfPath, lemmatizza, correggi, imgName):
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
        self.stop = None
        self.sub = re.compile(r"<[^\S>]+>|[^A-Za-z@]|\S+@\S+|[^\w\s]|http+\S+|www+\S+")
        self.path = dfPath
        self.cont = 0
        global c
        self.df = self.read_sample_astypeU(dfPath=dfPath,frac=frac)
        self.good_reviews = None
        self.bad_reviews = None
        self.count = None
        self.count_good = None
        self.count_bad = None
        self.count_good_words = None
        self.count_bad_words = None
        self.count_good_punctuations = None
        self.count_bad_punctuations = None
        c = c + 1
        print(f"[{c}]-Ricerca duplicati nel dataset...")
        describe = self.df.describe()
        c = c + 1
        if describe.iloc[1]['review'] != describe.iloc[0]['review']:
            print(f"[{c}]-Duplicati trovati pulizia in corso...")
            self.df.drop_duplicates(inplace=True)
            c = c + 1
            print(f"[{c}]-Pulizia terminata !")
        else:
            print(f"[{c}]-Duplicati non trovati !")
        self.option_subdivide_df()
        c = c + 1
        if correggi or lemmatizza or newProcessing:
            if lemmatizza:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            if newProcessing:
                print(f"[{c}]-Preparazione array di regex in corso...")
                self.present, self.sub1 = self.setArrayRegex()
                c = c + 1
                print(f"[{c}]-Preparazione array di regex completata !")
            self.chose_action_multiprocessing(correggi, lemmatizza, newProcessing)
            c = c + 1
        elif lemmatizza is False and correggi is False:
            print("Si vogliono aggiungere le attuali words_stop ?\nY/N")
            decision = input().lower()
            if decision == 'y':
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
                nome = input("Inserire nome nuovo file da creare (.csv escluso) : \n")
                file.to_csv(f"../Dataset_processed/{subdirectory}/{nome}.csv", index=False)
                c = c + 1
                print(f"[{c}]-File: {times}no_stop.csv salvato in ../Dataset_processed/{subdirectory} !")
        self.colorFrame = 'white'
        if imgName is not None:
            c = c + 1
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
            c = c + 1
            print(f"[{c}]-Selezione della maschera per word-cloud completata !")

    # FUNZIONI UTILI ALLA LETTURA, CAMPIONAMENTO, SETTAGGIO E CONVERSIONE DEL DATASET___________________________________

    def read_sample_astypeU(self, dfPath, frac):
        df = pd.read_csv(dfPath)
        df = df.sample(frac=frac)
        df = df.astype('U')
        return df

    def setDf(self, df):
        """
        Aggiorna il dataframe con il valore passato
        """
        self.df = df

    # FUNZIONI UTILI ALL'ESSTAPOLAZIONI DI INFORMAZIONE DEL DATASET_____________________________________________________

    def cal_len(self, data):
        """
        ritorna la lunghezza del dato passato
        """
        if isinstance(data, float):
            return 0
        return len(data)

    def printShapeOfDf(self):
        """
        stampa quante review negative e positive ha attualmente il dataset
        """
        print("The Shape of the Dataset".format(), self.df.shape)

    def option_subdivide_df(self):
        global c
        c = c + 1
        if self.option == '1':
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
            c = c + 1
            print(f'[{c}]-Verifico che il dataset sia bilanciato...')
            c = c + 1
            if len(self.count_good) != len(self.count_bad):
                # random under sampler è un modo semplice e veloce per generare un dataframe rappresentato da un
                # sottoinsieme del dataset di partenza ed elimina in modo randomico alcune review in base al target
                # che appare più volte
                rus = RandomUnderSampler(random_state=24)
                c = c + 1
                print(f"[{c}]-Dataset sbilanciato correzione in corso...")
                self.df, self.df['sentiment'] = rus.fit_resample(self.df, self.df['sentiment'])
                self.setDf(self.df)
                self.count_good = self.df[self.df['sentiment'] == 'positive']
                self.count_bad = self.df[self.df['sentiment'] == 'negative']
                c = c + 1
                print(f"[{c}]-Positive: {len(self.count_good)} - Negative: {len(self.count_bad)}")
                print(f"[{c}]-Dataset bilanciato !")
            else:
                print(f"[{c}]-Dataset bilanaciato azione non necessaria !")
        if self.option == '2':
            print(f"[{c}]-Divisione di soli elementi negativi per l'analisi in corso...")
            self.bad_reviews = self.df[self.df['sentiment'] == 'negative']['review']
            self.good_reviews = None
            self.count = self.bad_reviews.value_counts()
            self.count_good = None
            self.count_bad = self.df[self.df['sentiment'] == 'negative']
            self.count_good_words = None
            self.count_bad_words = self.count_bad['review'].str.split().apply(self.cal_len)
            self.count_good_punctuations = None
            self.count_bad_punctuations = self.count_bad['review'].apply(lambda z: len([c for c in str(z)
                                                                                        if c in
                                                                                        string.punctuation]))
            self.df = self.count_bad
            self.setDf(self.df)
            c = c + 1
            print(f"[{c}]-Divisione di soli elementi negativi per l'analisi completata !")
            c = c + 1
            target = 'negative'
            print(f'[{c}]-Le recensioni sono solo {target} !')

        if self.option == '3':
            print(f"[{c}]-Divisione di soli elementi positivi per l'analisi in corso...")
            self.bad_reviews = None
            self.good_reviews = self.df[self.df['sentiment'] == 'positive']['review']
            self.count = self.good_reviews.value_counts()
            self.count_good = self.df[self.df['sentiment'] == 'positive']
            self.count_bad = None
            self.count_good_words = self.count_good['review'].str.split().apply(self.cal_len)
            self.count_bad_words = None
            self.count_good_punctuations = self.count_good['review'].apply(lambda z: len([c for c in str(z)
                                                                                          if c in
                                                                                          string.punctuation]))
            self.count_bad_punctuations = None
            self.df = self.count_good
            self.setDf(self.df)
            c = c + 1
            print(f"[{c}]-Divisione di soli elementi positivi per l'analisi completata !")
            c = c + 1
            target = 'positive'
            print(f'[{c}]-Le recensioni sono solo {target} !')

    # FUNZIONI UTILI PER I PLOT DEL DATASET______________________________________________________________________________

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
                corpus = [i for x in self.good_reviews.str.split() for i in x]
                corpus.append(i for x in self.bad_reviews.str.split() for i in x)
            if self.option == '2':
                corpus = [i for x in self.bad_reviews.str.split() for i in x]
            if self.option == '3':
                corpus = [i for x in self.good_reviews.str.split() for i in x]
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

    # FUNZIONI UTILI PER IL SETTAGGIO DEI PARAMETRI DI MODIFICA DEL DATASET CON RELATIVE FUNZIONI DI MODIFICA___________

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                ________________________
                                                |                       |
                                                |        OPTION 2       |
                                                |_______________________|

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def setArrayRegex(self):
        """
        Viene utilizzata solo se scelgo di fare una prima pulizia del dataset nel main.
        Definisce e ritorna due array generati dal file regex che dovrà obbligatoriamente
        contenere una di queste due sitassi:
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
        Viene utilizzata solo se scelgo di fare una prima pulizia del dataset nel main
        self.present e self.sub1 sono due array che contengono lo stesso numero di elementi
        se viene trovata una espressione presente in self.present[i] o le sue alterative verrà sostituita con
        l'espressione presente in self.sub1[i] tramite sub del modulo re
        self.sub invece è un re.compile() che elimina linguaggio html, punteggiatura, siti, mail e altro
        """
        words = text.split(" ")
        words = [word for word in words if word != ' ' and word != '  ' and word != '    ' and word != '']
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
            i = i + 1

        words = [re.sub(self.sub, " ", word) for word in words]
        string = " ".join(word.strip() for word in words if word.strip())
        return string

    def proc_firstclean(self, df, results, cont, threadname, lock):
        global thread_name
        thread_name = "Start "+threadname
        print(thread_name)
        df['review'] = df['review'].apply(self.removeWithRe)
        thread_name = "Finish " + threadname
        print(thread_name)
        lock.acquire(block=True)
        results.append(df)
        lock.release()

    def multiproc_firstclean(self):
        lists_df = Manager().list()
        multi_proc = Multiprocessing(df=self.df, count=self.cal_len(self.df))
        list_df_sample = multi_proc.list_df
        list_proc = multi_proc.processes
        cpu = multi_proc.cpu
        lock = Lock()
        i = 0
        while i <= cpu - 1:
            threadname = "Process: " + str(i)
            list_proc[i] = Process(target=self.proc_firstclean, args=(list_df_sample[i], lists_df, i, threadname, lock))
            list_proc[i].start()
            i = i + 1
        i = 0
        while i <= cpu - 1:
            list_proc[i].join()
            i = i + 1

        dff = pd.concat(lists_df, axis=0)

        return dff
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                ________________________
                                                |                       |
                                                |        OPTION 5       |
                                                |_______________________|
                                                
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def correct(self, text):
        return str(TextBlob(text).correct())

    def proc_correct(self, df, results, cont, threadname, lock):
        global thread_name
        thread_name = "Start "+threadname
        print(thread_name)
        df['review'] = df['review'].apply(self.correct)
        thread_name = "Finish " + threadname
        print(thread_name)
        lock.acquire(block=True)
        results.append(df)
        lock.release()

    def multiproc_correct(self):
        lists_df = Manager().list()
        multi_proc = Multiprocessing(df=self.df, count=self.cal_len(self.df))
        list_df_sample = multi_proc.list_df
        list_proc = multi_proc.processes
        cpu = multi_proc.cpu
        lock = Lock()
        i = 0
        while i <= cpu - 1:
            threadname = "Process: " + str(i)
            list_proc[i] = Process(target=self.proc_correct, args=(list_df_sample[i], lists_df, i, threadname, lock))
            list_proc[i].start()
            i = i + 1
        i = 0
        while i <= cpu - 1:
            list_proc[i].join()
            i = i + 1

        dff = pd.concat(lists_df, axis=0)

        return dff

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   ________________________
                                                   |                       |
                                                   |        OPTION 3       |
                                                   |_______________________|

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
                if word != prev_word and prev_word != new_words[len(new_words) - 1]:
                    new_words.append(prev_word)
                    new_words.append(word)
                elif word != prev_word:
                    new_words.append(word)
                return new_words
            next_word = words[b]
            if not first:
                prev_word = words[a]
            if first is False and word != next_word and word != prev_word and c > 1 and word != new_words[
                len(new_words) - 1]:
                new_words.append(word)
            if first is False and word == next_word and word != prev_word:
                new_words.append(word)
            if first is False and word == prev_word and word != next_word:
                new_words.append(next_word)
            if first is True:
                new_words.append(word)

    def update_Stop(self, text):
        """
        Questo metodo viene chiamato quando decidiamo di analizzare un file e di aggiungerci le stop_words presenti in
        self.aggiornaStop_words
        """
        if text != 'nan':
            words = text.split(" ")
            words = [word.lower() for word in words if word.lower() not in self.stop]
            words = self.removeRepeat(words)
            string = " ".join(words)
            return string

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
        with open("../Stop_words/commoncopy.txt", 'r') as f:
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

        primo = True
        self.stop = set([word.lower() for word in stopwords])
        # self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        # self.df = pd.read_csv(self.path)
        print(f'\nSTART PROCESSING AT {datetime.now().strftime("%H:%M:%S")}')
        self.df = self.df.astype('U')
        self.df['review'] = self.df['review'].apply(self.update_Stop)
        print(f'\nFINISH PROCESSING AT {datetime.now().strftime("%H:%M:%S")}')
        # dimension = 49396
        # quattro da 6175 quattro da 6174
        self.setDf(self.df)

    def dumpVoc(self, arrayitem, arrayitem2, arrayitem3):
        stop = []
        with open('../Stop_words/films.txt', 'r') as f:
            s = f.readlines()
        [stop.append(x.rstrip()) for x in s]
        new_arrayvalues = []
        c = 0
        values = 0
        items = []
        for v in arrayitem3:
            items.append(v)
            items.append(arrayitem2[c])
            if c < int(len(arrayitem)):
                if arrayitem[c] not in stop:
                    items.append(arrayitem[c])
            c = c + 1
        items.sort()
        for v in items:
            new_arrayvalues.append(values)
            values = values + 1
        print(len(items))
        print(len(new_arrayvalues))
        vocabulary = {items[i]: new_arrayvalues[i] for i in range(len(items))}  # dict(zip(items, new_arrayvalues))
        with np.printoptions(threshold=np.inf):
            print(f"\n\nVOCABOLARIO: {vocabulary}\n\n")
        voc = input("Si vuole salvare questo vocabolario in venvServerAdmin2 ?\nY/N ?\n").lower()
        if voc == 'y':
            no = input("Inserire nome del file joblib da creare in venvServerAdmin2\n")
            print("Caricamento file in venvServerAdimin2 in corso ...")
            dump(value=vocabulary,
                 filename=f"../../modelAdmin/{no}.joblib")
            print("Caricamento file in venvServerAdimin2 completato !")

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   ________________________
                                                   |                       |
                                                   |        OPTION 4       |
                                                   |_______________________|

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def lemmatization(self, data):
        """
        Questa funzione viene chiamata quando decidiamo di lemmatizzare nel main e utilizzerà il modulo spacy
        en_core_web_sm
        """
        words = [token.lemma_ for token in self.nlp(data) if token.text != '' and token.text != '  ']
        string = " ".join(words)
        return string

    def proc_lemma(self, df, results, cont, threadname, lock):
        global thread_name
        thread_name = "Start "+threadname
        print(thread_name)
        df['review'] = df['review'].apply(self.lemmatization)
        thread_name = "Finish " + threadname
        print(thread_name)
        lock.acquire(block=True)
        results.append(df)
        lock.release()

    def multiproc_lemma(self):
        lists_df = Manager().list()
        multi_proc = Multiprocessing(df=self.df, count=self.cal_len(self.df))
        list_df_sample = multi_proc.list_df
        list_proc = multi_proc.processes
        cpu = multi_proc.cpu
        lock = Lock()
        i = 0
        while i <= cpu - 1:
            threadname = "Process: " + str(i)
            list_proc[i] = Process(target=self.proc_lemma, args=(list_df_sample[i], lists_df, i, threadname, lock))
            list_proc[i].start()
            i = i + 1
        i = 0
        while i <= cpu - 1:
            list_proc[i].join()
            i = i + 1

        dff = pd.concat(lists_df, axis=0)

        return dff

    def chose_action_multiprocessing(self, correggi, lemmatizza, newProcessing):
        global c

        if correggi is True:
            self.setDf(self.multiproc_correct())

        if lemmatizza is True:
            print(f"[{c}]-Lemmatizzazione avviata...")
            date_i = datetime.now().strftime("%H:%M:%S")
            self.setDf(self.multiproc_lemma())
            c = c + 1
            print(f'\n[{c}]-START AT {date_i}')
            print(f'\n[{c}]-FINISH AT {datetime.now().strftime("%H:%M:%S")}')
            print(f"[{c}]-Lemmatizzazione completata !")

        if newProcessing is True:
            c = c + 1
            print(f"[{c}]-Nuova pulizia avviata...")
            date_i = datetime.now().strftime("%H:%M:%S")
            self.setDf(self.multiproc_firstclean())
            c = c + 1
            print(f'\n[{c}]-START AT {date_i}')
            print(f'\n[{c}]-FINISH AT {datetime.now().strftime("%H:%M:%S")}')


