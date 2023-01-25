# Movie recommender system using Transformers based on Streamlit

# Install libraries
#!pip install kaggle
#!pip install zipfile


import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import os

from streamlit import session_state as session
from datetime import time, datetime
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer



###############################
## ------- FUNCTIONS ------- ##
###############################

#@st.cache(persist=True, show_spinner=False, suppress_st_warning=True)
@st.experimental_memo(persist=True, show_spinner=False, suppress_st_warning=True)
def load_dataset():
    """
    Load Dataset from Kaggle
    -return: dataframe containing dataset
    """
    # Downloading Movies dataset
    api.dataset_download_file('rounakbanik/the-movies-dataset', 'movies_metadata.csv')

    # Extract data
    zf = ZipFile('movies_metadata.csv.zip')
    zf.extractall() 
    zf.close()

    # Create dataframe
    data = pd.read_csv('movies_metadata.csv', low_memory=False)

    return data

def recommend_table(list_prefered_movies, movies_data, movie_count=10):
    """
    Function for recommending movies
    -param list_prefered_movies: list of movies selected by user
    -param tfidf_data: self-explanatory
    -param movie_count: number of movies to suggest
    -return: dataframe containing suggested movie
    """

    scores_similarity = []
    for movie in list_prefered_movies:
        a = query(
                {
                    "inputs": {
                        "source_sentence": movies_data[movies_data.title==movie],
                        "sentences": movies_data.overview.to_list(),
                    }
                }
            )


    movie_enjoyed_df = tfidf_data.reindex(list_of_movie_enjoyed)
    user_prof = movie_enjoyed_df.mean()
    tfidf_subset_df = tfidf_data.drop(list_of_movie_enjoyed)
    similarity_array = cosine_similarity(user_prof.values.reshape(1, -1), tfidf_subset_df)
    similarity_df = pd.DataFrame(similarity_array.T, index=tfidf_subset_df.index, columns=["similarity_score"])
    sorted_similarity_df = similarity_df.sort_values(by="similarity_score", ascending=False).head(movie_count)

    return sorted_similarity_df

def query(payload):
    """
    Get prediction from HuggingFace Inference API
    -param payload: json including the text to be compared
    -return: list of similarities
    """
    data = json.dumps(payload)
    API_URL = "https://api-inference.huggingface.co/models/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


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

# Create dataset
data = load_dataset()
#st.write(data[['title','overview']].head(3))


###############################
## ------ LOAD MODEL ------- ##
###############################

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')





###############################
## --------- MAIN ---------- ##
###############################

dataframe = None

st.title("""
Movie Recommendation System :film_frames:
This is a Content Based Recommender System based on movie synopsis :sunglasses:
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

#session.recommend_on = st.multiselect('Base recommendations on', ['Synopsis', 'Director', 'Genre', 'Duration'])
#st.text("")
#st.text("")

st.write('Base recommendations on:')
session.synopsis = st.checkbox('Synopsis')
session.director = st.checkbox('Director')
session.genre = st.checkbox('Genre')
session.duration = st.checkbox('Duration')

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend me a movie!")

st.write(data.iloc[:5].overview.to_list())

#if is_clicked:
#    dataframe = recommend_table(session.selected_movies, movies_data=data, movie_count=session.slider_count)

st.text("")
st.text("")
st.text("")
st.text("")

#if data is not None:
#    st.table(data)


