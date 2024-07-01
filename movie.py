import pickle
import streamlit as st
import requests
import re
import numpy as np 
from nltk.tokenize import word_tokenize

def preprocess(text):
  text = re.sub('[^a-zA-Z0-9]', ' ',text)
  tokens = word_tokenize(text.lower())
  return " ".join(tokens)

def hybrid_recommendation(movie_name):

  movie_name = str(movie_name).lower()  # Convert to lowercase for case-insensitive matching
  try:
    movie_id = np.where(pivot_table.reset_index(drop=False)["title"].apply(preprocess) == movie_name)[0][0]
    distance, suggestion = model.kneighbors(pivot_table.iloc[movie_id, :].values.reshape(1, -1), n_neighbors=6)
    for i in range(len(suggestion)):
            movies = pivot_table.index[suggestion[i]].tolist()
    try:
      idx = indices[movie_name]
      if idx is not None:
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        content_movies = content['title'].iloc[movie_indices].tolist()

    except IndexError:
      content_movies = []
    # Combine recommendations
    combined_movies = list(set(movies) | set(content_movies))

    print(f"You searched '{movie_name}'\n")

    if not combined_movies:
      print(f"No close matches found for '{movie_name}'")
    else:
       return combined_movies

  except IndexError:
    print(f"No close matches found for '{movie_name}'")


st.header('Movie Recommender System')


indices = pickle.load(open('indices.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
content = pickle.load(open('content.pkl','rb'))
pivot_table = pickle.load(open('pivot_table.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

movie_list = content['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = hybrid_recommendation(selected_movie)
   
    try:
        
     for i in range(len(recommended_movie_names)):
       st.text(recommended_movie_names[i])
        #st.image(recommended_movie_posters[0])
    except TypeError:
       st.write("No close matches found")    
   
   
