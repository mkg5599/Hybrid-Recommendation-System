import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1) IMPORTING ALL NECESSARY PACKAGES
# ----------------------------------------------------------------------
import requests
import numpy as np
import pandas as pd
import time
import re
import math

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

from collections import Counter

# SciKit-Learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import TruncatedSVD

# For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout
plotly.offline.init_notebook_mode(connected=True)

# Others
import pickle

# For images
from io import BytesIO
from PIL import Image

# For Keras / TensorFlow based feature extraction
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# ----------------------------------------------------------------------
# 2) READ JSON DATA (Example: tops_fashion.json) / Then filter columns
#    (We assume you have the file or skip if not needed.)
# ----------------------------------------------------------------------
'''
data = pd.read_json('tops_fashion.json')
print("Number of data points :", data.shape[0])
print("Number of features :", data.shape[1])
data.columns
data.head()
'''

# We'll only keep relevant columns
#  If continuing from partial data of the conversation, skip or modify
# data = data[['asin','brand','color','medium_image_url','product_type_name','title','formatted_price']]

# Example further filtering
'''
data = data[~data['formatted_price'].isnull()]
data = data[~data['color'].isnull()]
print("Data shape after removing nulls:", data.shape)
'''

# Saving & loading pickles:
# data.to_pickle('pickels/28k_apparel_data')
# data = pd.read_pickle('pickels/28k_apparel_data')

# Further deduplicate steps or so. The user in conversation dropped many duplicates.

# ----------------------------------------------------------------------
# 3) TEXT PREPROCESSING (Stopword removal, etc.)
# ----------------------------------------------------------------------
stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text, index, column, df):
    # This function modifies the df[column][index] in place
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            word = "".join(e for e in words if e.isalnum())  # remove special chars
            word = word.lower()                             # to lowercase
            if word not in stop_words:
                string += word + " "
        df.at[index, column] = string.strip()

# Example usage:
'''
start_time = time.time()
for idx, row in data.iterrows():
    nlp_preprocessing(row['title'], idx, 'title', data)
print("Time taken for text-preprocessing:", time.time() - start_time)
'''

# ----------------------------------------------------------------------
# 4) TEXT-BASED PRODUCT SIMILARITY:
#    (a) Bag-of-Words
#    (b) TF-IDF
#    (c) IDF Weighted
#    (d) Word2Vec (average & weighted)
# ----------------------------------------------------------------------

# UTILITY PLOTTING / DISPLAY FUNCTIONS:

def display_img(url, ax, fig):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it in notebook
    plt.imshow(img)

def plot_heatmap(keys, values, labels, url, text):
    # keys : list of words of recommended title
    # values: occurrence of each word
    # labels: (depends on the model e.g. tf-idf values, etc.)
    # url: image url to display
    # text: recommended item text
    gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
    fig = plt.figure(figsize=(25,3))
    
    # 1) heatmap
    ax = plt.subplot(gs[0])
    ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
    ax.set_xticklabels(keys)
    ax.set_title(text)
    # 2) image
    ax = plt.subplot(gs[1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    display_img(url, ax, fig)
    plt.show()

def plot_heatmap_image(doc_id, vec1, vec2, url, text, model,
                       tfidf_vectorizer=None, tfidf_features=None,
                       idf_vectorizer=None, idf_features=None):
    # doc_id: index of the input doc
    # vec1: input apparel vector (dict: {word:count})
    # vec2: recommended apparel vector (dict: {word:count})
    # model: 'bag_of_words', 'tfidf', 'idf'

    # find common words
    intersection = set(vec1.keys()) & set(vec2.keys())
    for i in vec2:
        if i not in intersection:
            vec2[i] = 0
    
    keys = list(vec2.keys())
    values = [vec2[x] for x in vec2.keys()]
    
    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            if (tfidf_vectorizer is not None) and (x in tfidf_vectorizer.vocabulary_):
                labels.append(tfidf_features[doc_id, tfidf_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            if (idf_vectorizer is not None) and (x in idf_vectorizer.vocabulary_):
                labels.append(idf_features[doc_id, idf_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    plot_heatmap(keys, values, labels, url, text)

def text_to_vector(text):
    wordPattern = re.compile(r'\w+')
    words = wordPattern.findall(text)
    return Counter(words)

def get_result(doc_id, content_a, content_b, url, model,
               df, tfidf_title_vectorizer=None, tfidf_title_features=None,
               idf_title_vectorizer=None, idf_title_features=None):
    # content_a: input item text
    # content_b: recommended item text
    vector1 = text_to_vector(content_a)
    vector2 = text_to_vector(content_b)
    
    plot_heatmap_image(doc_id, vector1, vector2, url, content_b, model,
                       tfidf_title_vectorizer, tfidf_title_features,
                       idf_title_vectorizer, idf_title_features)


# ------------------- BAG OF WORDS -----------------------
def bag_of_words_model(doc_id, num_results, data, title_features, title_vectorizer):
    # doc_id in data's index
    pairwise_dist = pairwise_distances(title_features, title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        get_result(indices[i],
                   data['title'].loc[df_indices[0]],
                   data['title'].loc[df_indices[i]],
                   data['medium_image_url'].loc[df_indices[i]],
                   'bag_of_words',
                   data)
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("Brand:", data['brand'].loc[df_indices[i]])
        print("Title:", data['title'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*60)

# ------------------- TF-IDF -----------------------------
def tfidf_model(doc_id, num_results, data, tfidf_title_features, tfidf_title_vectorizer):
    pairwise_dist = pairwise_distances(tfidf_title_features, tfidf_title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        get_result(indices[i],
                   data['title'].loc[df_indices[0]],
                   data['title'].loc[df_indices[i]],
                   data['medium_image_url'].loc[df_indices[i]],
                   'tfidf',
                   data,
                   tfidf_title_vectorizer, tfidf_title_features)
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("BRAND:", data['brand'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)

# ------------------- IDF Weighted ------------------------
def n_containing(word, df):
    return sum(1 for blob in df['title'] if word in blob.split())

def idf(word, df):
    return math.log(df.shape[0] / (n_containing(word, df)))

def idf_model(doc_id, num_results, data, idf_title_features):
    # idf_title_features we built by counting each word, then manually weighting with idf
    pairwise_dist = pairwise_distances(idf_title_features, idf_title_features[doc_id])
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        get_result(indices[i],
                   data['title'].loc[df_indices[0]],
                   data['title'].loc[df_indices[i]],
                   data['medium_image_url'].loc[df_indices[i]],
                   'idf',
                   data)
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("Brand :", data['brand'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)


# ----------------------------------------------------------------------
# 5) WORD2VEC SEMANTICS
#    (a) Average W2V
#    (b) Weighted (IDF) W2V
# ----------------------------------------------------------------------
# Suppose we have a loaded word2vec model. (We used a variable `model` or `vocab`.)
# For demonstration, we skip the actual Gensim load. You must do something like:
'''
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocab = model.key_to_index  # in modern Gensim
'''

def get_word_vec(sentence, doc_id, m_name, model, vocab, idf_title_vectorizer, idf_title_features):
    # m_name can be 'avg' or 'weighted'
    vec = []
    for i in sentence.split():
        if i in vocab:
            if m_name=='weighted' and (idf_title_vectorizer is not None) and (i in idf_title_vectorizer.vocabulary_):
                weight = idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[i]]
                vec.append(weight * model[i])
            elif m_name=='avg':
                vec.append(model[i])
        else:
            # word not in model vocabulary
            vec.append(np.zeros((300,)))
    return np.array(vec)

def get_distance(vec1, vec2):
    final_dist = []
    for i in vec1:
        dist = []
        for j in vec2:
            dist.append(np.linalg.norm(i-j))
        final_dist.append(np.array(dist))
    return np.array(final_dist)

def heat_map_w2v(sentence1, sentence2, url, doc_id1, doc_id2, df_id1, df_id2,
                 model, data, w2vMode, 
                 idf_title_vectorizer=None, idf_title_features=None, vocab=None):
    # build vectors:
    s1_vec = get_word_vec(sentence1, doc_id1, w2vMode, model, vocab,
                          idf_title_vectorizer, idf_title_features)
    s2_vec = get_word_vec(sentence2, doc_id2, w2vMode, model, vocab,
                          idf_title_vectorizer, idf_title_features)
    s1_s2_dist = get_distance(s1_vec, s2_vec)
    
    # Plot using matplotlib:
    gs = gridspec.GridSpec(2,2, width_ratios=[4,1], height_ratios=[2,1])
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(gs[0])
    ax = sns.heatmap(np.round(s1_s2_dist,4), annot=True)
    ax.set_xticklabels(sentence2.split())
    ax.set_yticklabels(sentence1.split())
    ax.set_title(sentence2)
    
    ax = plt.subplot(gs[1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    display_img(url, ax, fig)
    
    plt.show()


# EXAMPLE:
# We'll store final w2v embeddings of each doc in w2v_title or w2v_title_weight arrays for distance calculations.

def build_avg_vec(sentence, num_features, doc_id, w2vMode, model, vocab,
                  idf_title_vectorizer=None, idf_title_features=None):
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0
    for word in sentence.split():
        if word in vocab:
            nwords += 1
            if w2vMode=='weighted' and (idf_title_vectorizer is not None) and (word in idf_title_vectorizer.vocabulary_):
                weight_val = idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]]
                featureVec = np.add(featureVec, weight_val*model[word])
            elif w2vMode=='avg':
                featureVec = np.add(featureVec, model[word])
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def avg_w2v_model(doc_id, num_results, data, w2v_title):
    # doc_id is integer index
    pairwise_dist = pairwise_distances(w2v_title, w2v_title[doc_id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        # we can do a small heat map or so
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("Brand:", data['brand'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)

def weighted_w2v_model(doc_id, num_results, data, w2v_title_weight):
    pairwise_dist = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("Brand:", data['brand'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)

# ----------------------------------------------------------------------
# 6) BRAND / COLOR Features + Combined Weighted Distances
# ----------------------------------------------------------------------
# If brand or color has spaces, we can do
# brand = brand.replace(' ','-')
# Then create a CountVectorizer. For example:
'''
brands = [ str(x).replace(' ','-') for x in data['brand'] ]
types  = [ str(x).replace(' ','-') for x in data['product_type_name'] ]
colors = [ str(x).replace(' ','-') for x in data['color'] ]

brand_vectorizer = CountVectorizer()
brand_features = brand_vectorizer.fit_transform(brands)

type_vectorizer = CountVectorizer()
type_features = type_vectorizer.fit_transform(types)

color_vectorizer = CountVectorizer()
color_features = color_vectorizer.fit_transform(colors)

from scipy.sparse import hstack
extra_features = hstack((brand_features, type_features, color_features)).tocsr()
'''

def heat_map_w2v_brand(sentance1, sentance2, url, doc_id1, doc_id2, 
                       df_id1, df_id2, model, data, w2vMode,
                       brand_color_info, # e.g. brand/color values to display
                       idf_title_vectorizer=None, idf_title_features=None, vocab=None):
    """
    brand_color_info: 
      some data structure or the actual brand, color from data
      here we skip certain details for simplicity
    """
    # For brevity, the user can incorporate brand in a table
    # We'll do a partial example
    pass

def idf_w2v_brand(doc_id, w1, w2, num_results, data, 
                  w2v_title_weight, extra_features):
    # w1 -> weight for w2v
    # w2 -> weight for brand/color
    w2v_dist = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1,-1))
    exfeat_dist = pairwise_distances(extra_features, extra_features[doc_id])
    pairwise_dist = (w1*w2v_dist + w2*exfeat_dist)/(float(w1+w2))
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    
    for i in range(len(indices)):
        print("ASIN :", data['asin'].loc[df_indices[i]])
        print("Brand:", data['brand'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)

# If we also want CNN feature weighting, we do something like:
def idf_w2v_brand_cnn(doc_id, w1, w2, w3, num_results, data,
                      w2v_title_weight, extra_features, bottleneck_features_train):
    w2v_dist  = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1,-1))
    exfeat_dist = pairwise_distances(extra_features, extra_features[doc_id])
    cnn_dist  = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))
    pairwise_dist = (w1*w2v_dist + w2*exfeat_dist + w3*cnn_dist)/(float(w1+w2+w3))
    
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])
    for i in range(len(indices)):
        print("ASIN:", data['asin'].loc[df_indices[i]])
        print("Distance:", pdists[i])
        print("="*125)

# ----------------------------------------------------------------------
# 7) CNN FEATURE EXTRACTION (VGG16) for visual similarity
# ----------------------------------------------------------------------
# Example code snippet to compute VGG16 "bottleneck" features for images in a directory.
'''
img_width, img_height = 224, 224
train_data_dir = "images2/"
nb_train_samples = 16042
batch_size = 1

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)
    model = applications.VGG16(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    asins = []
    for i in generator.filenames:
        asins.append(i[2:-4])  # example slicing
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples//batch_size)
    bottleneck_features_train = bottleneck_features_train.reshape((16042,25088))
    np.save(open('16k_data_cnn_features.npy','wb'), bottleneck_features_train)
    np.save(open('16k_data_cnn_feature_asins.npy','wb'), np.array(asins))
'''

# Then we can load them:
# bottleneck_features_train = np.load('16k_data_cnn_features.npy')
# asins = np.load('16k_data_cnn_feature_asins.npy')

# A function to get similar products based on CNN features alone:
def get_similar_products_cnn(doc_id, num_results,
                             data, asins, df_asins,
                             bottleneck_features_train):
    # doc_id from data's index -> must map to the index in asins
    # we do asins.index(...) or so
    # example:
    doc_asin = df_asins[doc_id]
    doc_asin_index = asins.tolist().index(doc_asin)
    
    pairwise_dist = pairwise_distances(bottleneck_features_train, 
                                       bottleneck_features_train[doc_asin_index].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[:num_results]
    
    for i in range(len(indices)):
        # find that i's asin in the asins array:
        rec_asin = asins[indices[i]]
        # find row in data
        row_index = data[data['asin']==rec_asin].index
        # if multiple matches, take the first
        # ...
        # Then display
        print("ASIN:", rec_asin)
        print("Distance from input image:", pdists[i])
        print("Url: www.amazon.com/dp/%s" % rec_asin)
        print("-"*50)

# ----------------------------------------------------------------------
# 8) COLLABORATIVE FILTERING EXAMPLE
# ----------------------------------------------------------------------
# Suppose we have a user rating data '16kcollab.csv'
def collaborative_example():
    amazon_ratings = pd.read_csv('16kcollab.csv')
    amazon_ratings = amazon_ratings.dropna()
    popular_products = pd.DataFrame(amazon_ratings.groupby('productid')['rating'].count())
    most_popular = popular_products.sort_values('rating', ascending=False)
    
    # We create a utility matrix:
    amazon_ratings1 = amazon_ratings.head(100000)
    ratings_utility_matrix = amazon_ratings1.pivot_table(values='rating',
                                                         index='userid',
                                                         columns='productid',
                                                         fill_value=0)
    X = ratings_utility_matrix.T  # product on rows, user on columns
    print("Shape of X:", X.shape)
    # SVD to reduce dimensionality
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    # Then pick an item i
    i = "B0085N3P2E" # example
    product_names = list(X.index)
    product_ID = product_names.index(i)
    correlation_product_ID = correlation_matrix[product_ID]
    

# ----------------------------------------------------------------------
# MAIN or TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Loaded single script with combined code from the conversation.")
    print("Use the functions as needed. For example:")
    # e.g. data = pd.read_pickle('pickels/16k_apparel_data_preprocessed')

    # Then fit bag-of-words on 'title':
    # title_vectorizer = CountVectorizer()
    # title_features = title_vectorizer.fit_transform(data['title'])
    # bag_of_words_model(doc_id=12566, num_results=10, data=data,
    #                    title_features=title_features,
    #                    title_vectorizer=title_vectorizer)

    print("All major functions are declared.")
    print("You can call them in a sequence relevant to your pipeline.")
