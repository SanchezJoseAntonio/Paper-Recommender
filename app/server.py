from fastapi import FastAPI
import joblib
from sklearn.neighbors import NearestNeighbors
import pandas as pd

df = joblib.load('./app/papers_df.joblib')
vectorizer = joblib.load('./app/vectorizer.joblib')
nn = joblib.load('./app/nearest_neighbors.joblib')

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message':'Paper Recommender API'}

@app.post('/recommend')
def recommend_with_scores(query):
    """
    Gives the top 3 most similar papers (based on abstract) to the input.
    Args:
        query (string): A string containing the desired words
        e.g "Machine learning and deep learning"
    Returns:
        list: A list containing the titles of the top 3 most similar papers 
        and their score.
    """
    q_vec = vectorizer.transform([query]) 
    distances, indices = nn.kneighbors(q_vec, n_neighbors=3)
    scores = 1 - distances[0]  # cosine similarity
    results = [(idx, float(score)) for idx, score in zip(indices[0], scores)]
    recommendations = []
    for index, score in results:
        recommendations.append(f"{df[index]} — score: {score:.4f}") # Show the title instead of the abstract (easier for search)
    return recommendations