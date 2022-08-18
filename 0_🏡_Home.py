import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_ace import st_ace
from joblib import load
import random
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup as bs
import math
from torchtext.vocab import GloVe

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout='wide')

# Create a sidebar and populate with content
st.sidebar.title("About")
st.sidebar.info(
    """
    [Web App URL](https://inspiritai-tech-demo-fake-news-detection-0--home-2r2yw7.streamlitapp.com/)
    \n
    [GitHub Respository](https://github.com/inspiritai-tech/demo-fake-news-detection)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Inspirit AI: <https://www.inspiritai.com>
    [GitHub](https://github.com/inspiritai-tech) | [LinkedIn](https://www.linkedin.com/company/inspirit-ai/)
    """
)

# Title
st.title("📰 Fake News Detection Powered by Neural Networks")
st.subheader("Making Sense of the Age of Misinformation")
st.markdown(
    '''
    We live in an age of rampant disinformation, both deliberate and accidental. The proliferation of
    fake news has been accelerated in recent years through social media channels like Facebook, What's App,
    Reddit, and others. Whether deliberate or accidental, misinformation drives powerful social and psychological
    forces that shape how we behave in our society.
    '''
)

# Hero Image
st.image("hero-image.jpg")
st.caption("Photo Credit: [Ludovica Dri](https://unsplash.com/@wanderluly?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/Bc_y35IwUHw). ")

st.markdown(
    '''
    High quality, verfied information is the bedrock of a healthy public so that individuals can make
    informed decisions about their own lives. Spotting fake news, however, can be a particularly tricky task.
    Even a keen-eyed individual is not immune to the subtleties of unverified information. As a light example,
    take the following news headline:
    '''
)

st.warning("\"CAT LOVERS CAN TRY CAT FOOD INSPIRED DISHES AT FANCY FEAST'S ITALIAN POP-UP\"")

st.markdown(
    '''
    There's nothing that you can exactly *point to* that's wrong with this headline, other than it feels absurd.
    You may even be surprised to find that it's a [real headline](https://www.cnn.com/2022/07/31/business/fancy-feast-cat-food-restaurant-trnd/index.html).
    Often time, the news is happening around is so fast and is spread widely online in so many different
    formats that our attention might be so exhausted that we take what we see as fact.
    '''
)

st.header("How AI Can Help Detect Fake News")

st.markdown(
    '''
    If we train a machine learning model on as many examples of fake and real news articles as possible, then
    we can deploy this tool to filter out fake content on information platforms before they proliferate enough to
    substantially influence public opinion.

    To learn more about how this works under the hood, keep reading this article. First, however, feel free to play
    around with our fake news detection tool by plugging in any website to get a prediction for what our AI thinks of
    that article.
    '''
)

st.markdown('---')

# AI Fake News Demo
st.header(" 🗞️ Try: AI Fake News Detector 📰 ")

model = pickle.load(open('fake_news_model.sav', 'rb'))

news_site = st.text_input(
    '''
    Paste in any link in the space below to recieve an AI powered prediction for whether your site is fake news or bona fide!
    '''
)

VEC_SIZE = 300

glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def get_data_pair(url):
    if not url.startswith('http'):
        url = 'http://' + url
    url_pretty = url
    if url_pretty.startswith('https://'):
        url_pretty = url_pretty[7:]
    if url_pretty.startswith('https://'):
        url_pretty = url_pretty[8:]
    response = requests.get(url, timeout=10)
    htmltext = response.text

    return url_pretty, htmltext

def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X

def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))

def keyword_featurizer(url, html):
    features = {}
    
    # Same as before.
    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')
    
    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance']
    
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)
    
    return features

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split(): 
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words        
    return X

vectorizer = CountVectorizer(max_features=300)

train_descriptions = pickle.load(open('train_descriptions', 'rb'))

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
    description = ''
  return description

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

def featurize_data_pair(url, html):
  # Approach 1.
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  # Approach 2.
  description = get_description_from_html(html)
  
  bow_X = vectorize_data_descriptions([description], vectorizer)
  
  # Approach 3.
  glove_X = glove_transform_data_descriptions([description])
  
  X = combine_features([keyword_X, bow_X, glove_X])
  
  return X

clicked = st.button("Generate Prediction")

if clicked:
    url, html = get_data_pair(news_site)
    curr_X = featurize_data_pair(url, html)
    curr_y = model.predict(curr_X)[0]
    if curr_y < 0.5:
        st.balloons()
        st.write("You submitted: " + news_site)
        st.subheader("🥳 The AI predicts that this site is real news! 🦾")
    else:
        st.write("You submitted: " + news_site)
        st.subheader("🤡 The AI predicts that this site is fake news! 👻")

# if clicked:
#     num = random.randint(1, 2)
#     if num == 1:
#         st.balloons()
#         st.write("You submitted: " + news_site)
#         st.subheader("🥳 The AI predicts that this site is real news! 🦾")
#     else:
#         st.write("You submitted: " + news_site)
#         st.subheader("🤡 The AI predicts that this site is fake news! 👻")

st.markdown('---')

st.image("newspapers.jpg")
st.caption("Photo Credit: [Fabian Barral](https://unsplash.com/@iammrcup?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/photos/Mwuod2cm8g4). ")

st.header("How Does this Work?")
st.subheader("Anatomy of a News Website")
st.markdown(
    '''
    Have you ever wondered how websites like [google.com](https://www.google.com) and [nytimes.com](https://www.nytimes.com) work under the hood?
    Using the internet every day, it is easy to forget how magical even the most mundane web browsing experiences
    are. 

    The answer is HTML which stands for Hyper Text Markup Language that describes the structure of Web pages
    in elemental block form. To play around with HTML, open up this [interactive environment](https://www.w3schools.com/html/tryit.asp?filename=tryhtml_default)
    to play around with writing your own HTML code! Additionally, if you go to any site, if you right click on the page, you'll be able to
    'View Source' or 'View Selection' to see the HTML source code for that page.

    Now that we know how to extract the core information that encodes any given news site, we can take the URL of a news site
    along with its HTML source code to train our machine learning model on how to differentiate between real and fake news.
    '''
)