# Movie recommender system using Transformers based on Streamlit

# Install libraries
#!pip install kaggle
#!pip install zipfile


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import os

from streamlit import session_state as session
from datetime import time, datetime
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer


###############################
## --- CONNECT TO KAGGLE --- ##
###############################

# Authenticate Kaggle account
os.environ['KAGGLE_USERNAME'] = st.secrets['username']
os.environ['KAGGLE_KEY'] = st.secrets['key']


api_token = {"username":st.secrets['username'],"key":st.secrets['key']}
with open('/home/appuser/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)


# Activate Kaggle API
api = KaggleApi()
api.authenticate()


###############################
## ----- OBTAIN DATASET ---- ##
###############################

# Downloading Movies dataset
api.dataset_download_file('rounakbanik/the-movies-dataset', 'movies_metadata.csv')

# Extract data
zf = ZipFile('movies_metadata.csv.zip')
zf.extractall() 
zf.close()

# Show first rows of dataset
data = pd.read_csv('movies_metadata.csv', low_memory=False)
#st.write(data[['title','overview']].head(3))


###############################
## ------ LOAD MODEL ------- ##
###############################

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')





###############################
## ------ APP MAIN --------- ##
###############################

dataframe = None

st.title("""
Movie Recommendation System :film_frames:
This is a Content Based Recommender System based on movie synopsis :sunglasses:.
 """)

st.text("")
st.text("")
st.text("")
st.text("")

session.selected_movies = st.multiselect(label="Select prefered movies", options=data.title)

st.text("")
st.text("")

session.slider_count = st.slider(label="Number of results", min_value=3, max_value=10)

st.text("")
st.text("")

session.recommend_on = st.multiselect('Base recommendations on', ['Synopsis', 'Director', 'Genre', 'Duration'])

st.text("")

st.write('Base recommendations on:')
session.synopsis = st.checkbox('Synopsis')
session.director = st.checkbox('Director')
session.genre = st.checkbox('Genre')
session.duration = st.checkbox('Duration')

st.text("")
st.text("")

#buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

#is_clicked = col1.button(label="Make recommendations")

#if is_clicked:
#    dataframe = recommend_table(session.selected_movies, movie_count=session.slider_count, tfidf_data=tfidf)

st.text("")
st.text("")
st.text("")
st.text("")

#if data is not None:
#    st.table(data)


