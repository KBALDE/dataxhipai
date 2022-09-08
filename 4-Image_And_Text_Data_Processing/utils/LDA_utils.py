# some utils functions for LDA
import pandas as pd
import numpy as np

import sys
from pprint import pprint
import re

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
# from gensim.utils import lemmatize, simple_preprocess cant't import lemmatize from gensim
# we can get from nltk
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say',
                   'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try',
                   'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 
                   'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 
                   'even', 'also', 'may', 'take', 'come'])

# tokenize using gensim
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent) 
        


# T_SNE 
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

def plotT_SNE(lda_model, corpus):
    
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])
    
    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    
    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]
    
    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)
    
    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    
    # Plot the Topic Clusters using Bokeh
    output_notebook()
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    show(plot)
    
    
# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

def plotWordCloud(lda_model):
    
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    
    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    
    topics = lda_model.show_topics(formatted=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
    
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()

    