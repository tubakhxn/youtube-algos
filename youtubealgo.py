# Install Surprise library
# pip install scikit-surprise

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample data (user_id, video_id, rating)
data = [
    ('user1', 'video1', 5),
    ('user1', 'video2', 4),
    ('user2', 'video1', 3),
    ('user2', 'video3', 5),
    ('user3', 'video2', 2),
    # Add more data as needed
]

# Define the reader object to parse the data
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise dataset
dataset = Dataset.load_from_df(data, reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Use the KNNBasic collaborative filtering algorithm
sim_options = {
    'name': 'cosine',
    'user_based': True  # User-based collaborative filtering
}

algo = KNNBasic(sim_options=sim_options)

# Train the algorithm on the training set
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate the model performance
accuracy.rmse(predictions)

# Function to get video recommendations for a given user
def get_recommendations(user_id, n=5):
    # Get a list of all video IDs
    all_video_ids = list(set([video_id for _, video_id, _ in data]))

    # Remove videos the user has already watched
    watched_videos = set([video_id for _, video_id, _ in data if user_id == _])
    to_be_predicted = list(watched_videos.symmetric_difference(all_video_ids))

    # Predict ratings for videos the user hasn't watched
    predicted_ratings = [algo.predict(user_id, video_id).est for video_id in to_be_predicted]

    # Combine video IDs with predicted ratings
    video_ratings = list(zip(to_be_predicted, predicted_ratings))

    # Sort the videos by predicted rating in descending order
    video_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get the top N recommendations
    top_recommendations = video_ratings[:n]

    return top_recommendations
