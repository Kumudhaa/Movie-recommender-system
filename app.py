from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from nltk.tokenize import word_tokenize

indices = pickle.load(open('indices.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
content = pickle.load(open('content.pkl','rb'))
pivot_table = pickle.load(open('pivot_table.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def preprocess(text):
  text = re.sub('[^a-zA-Z0-9]', ' ',text)
  tokens = word_tokenize(text.lower())
  return " ".join(tokens)

def hybrid_recommendation(movie_name):

  movie_name = str(movie_name).lower()  # Convert to lowercase for case-insensitive matching
  try:
    movie_id = np.where(pivot_table.reset_index(drop=False)["title"].apply(preprocess) == movie_name)[0][0]
    distance, suggestion = model.kneighbors(pivot_table.iloc[movie_id, :].values.reshape(1, -1), n_neighbors=6)
    movies = pivot_table.index[suggestion[0]].tolist()

    try:
      idx = indices[movie_name]
      if idx is not None:
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        content_movies = content['title'].iloc[movie_indices].tolist()
      else:
        content_movies = []
    except IndexError:
      content_movies = []

    combined_movies = list(set(movies) | set(content_movies))

    return combined_movies

  except IndexError:
    return []  # Handle case where no close matches are found

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def recommend_movies():
  movie_list = content['title'].values
  selected_movie = None
  recommended_movie_names = []

  if request.method == 'POST':
    selected_movie = request.form['movie_name']
    if selected_movie:
      recommended_movie_names = hybrid_recommendation(selected_movie)

  return render_template('home.html', movie_list=movie_list, selected_movie=selected_movie,
                         recommended_movie_names=recommended_movie_names)

if __name__ == '__main__':
  app.run(debug=True)