"""

file ini adalah rekomendasi wisata berdasarkan hasil onboarding user ketika memilih kategori di awal aplikasi

"""

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load the data
df = pd.read_csv(r'C:\Users\kasmi\Documents\GitHub\testing-API\user_rating_clean.csv')

# Train the item similarity model
train_data, _ = train_test_split(df, test_size=0.2, random_state=42)
train_item_user_matrix = train_data.pivot_table(index='Place_Id', columns='User_Id', values='Place_Ratings', fill_value=0)
item_similarity_train = cosine_similarity(train_item_user_matrix)
item_similarity_model = pd.DataFrame(item_similarity_train, index=train_item_user_matrix.index, columns=train_item_user_matrix.index)

# Load the item similarity model
# Memuat model dan data
item_similarity_model = pd.read_pickle(r'C:\Users\kasmi\Documents\GitHub\testing-API\fitur3.pkl')


def recommend_places(category, item_similarity_model, top_n=10):
    # Filter places based on the category
    places_in_category = df[df['Category'] == category]['Place_Id'].unique()
    valid_places_in_category = set(places_in_category) & set(item_similarity_model.columns)

    if not valid_places_in_category:
        return pd.DataFrame(columns=['Place_Id', 'Place_Name', 'Category'])

    place_similarity_scores = item_similarity_model[list(valid_places_in_category)].mean(axis=1)
    top_places = place_similarity_scores.sort_values(ascending=False).index
    recommended_places = df[df['Place_Id'].isin(top_places) & (df['Category'] == category)][['Place_Id', 'Place_Name', 'Category', 'City']].drop_duplicates()

    return recommended_places.head(top_n)

@app.route('/', methods=['GET'])
def home():
    return "Selamat datang di Tourism Recommendation API!"

@app.route('/onboarding', methods=['POST'])
def recommend_places_api():
    data = request.get_json()

    if 'category' not in data:
        return jsonify({'error': 'Category is required'}), 400

    category = data['category'].lower()
    df['Category'] = df['Category'].str.lower()

    valid_categories = df['Category'].unique()

    if category not in valid_categories:
        return jsonify({'error': 'Invalid category'}), 400

    recommended_places = recommend_places(category, item_similarity_model, top_n=10)

    if recommended_places.empty:
        return jsonify({'message': 'No recommendations for the given category'}), 200
    else:
        return jsonify({'recommended_places': recommended_places.to_dict(orient='records')}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
