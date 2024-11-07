import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Example job descriptions or corpus (add more if you have multiple job descriptions)
job_description = """
We are looking for a Data Scientist with experience in machine learning, Python, and SQL. 
The candidate should have expertise in data preprocessing, model development, and deployment.
Preferred skills include TensorFlow, scikit-learn, and cloud technologies like AWS or Azure.
"""

# Training the TF-IDF Vectorizer model
def train_tfidf_model(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    return vectorizer

# Create a list of job descriptions or any relevant text corpus
corpus = [job_description]

# Train the TF-IDF model on the job description
vectorizer = train_tfidf_model(corpus)

# Save the trained model as a .pkl file
with open("models/tfidf_model.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("TF-IDF model saved as tfidf_model.pkl")
