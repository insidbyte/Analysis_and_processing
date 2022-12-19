from collections import defaultdict
import pandas as pd
import plotly.graph_objs as go


class Analyses_2:
    def __init__(self, ngrams, df):
        self.ngrams = ngrams
        self.df = df
        self.freqDict = None

    def setFreqDict(self, freq_dict):
        self.freqDict = freq_dict

    def gram_analysis(self, data):
        data = [t for t in data.lower().split(" ") if t != ""]
        ngrams = zip(*[data[i:] for i in range(self.ngrams)])
        final_tokens = [" ".join(z) for z in ngrams]
        return final_tokens

    def horizontal_bar_chart(self, df, color):
        trace = go.Bar(
            y=df["n_gram_words"].values[::-1],
            x=df["n_gram_frequency"].values[::-1],
            showlegend=False,
            orientation='h',
            marker=dict(
                color=color,
            ),
        )
        return trace

    def createDict(self, data):
        data = data['review']
        freq = defaultdict(int)
        for d in data:
            for token in self.gram_analysis(d):
                freq[token] += 1
        self.setFreqDict(freq_dict=freq)
        return freq

    def create_new_df(self, n):
        freq_df = pd.DataFrame(sorted(self.freqDict.items(), key=lambda z: z[1])[::-1])
        freq_df.columns = ['n_gram_words', 'n_gram_frequency']
        trace = self.horizontal_bar_chart(freq_df[:n], 'orange')
        return trace


