# Movie recommender system using Transformers based on Streamlit

# Install libraries
!pip install kaggle
!pip install zipfile


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import time, datetime
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

# Activate Kaggle API
api = KaggleApi()
api.authenticate()


# Downloading Movies dataset
api.dataset_download_file('rounakbanik/the-movies-dataset', 'movies_metadata.csv')

# Extract data
zf = ZipFile('movies_metadata.csv.zip')
zf.extractall() 
zf.close()

# Show first rows of dataset
data = pd.read_csv('movies_metadata.csv')
st.write(data.head(5))
