import nltk

# do this only once
# nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Function to calculate sentiment score using NLTK's VADER
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0  # Assign neutral sentiment score if post is empty
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score is a single value representing overall sentiment

def feature_engineering_with_nltk(df):
    # Calculate social media account age (just a mock function for demonstration)
    def calculate_account_age(created_date):
        # Assume current year is 2024
        return 2024 - pd.to_datetime(created_date).year
    
    # Calculate social media account age
    df['fb_age'] = df['facebook_handle_created_date'].apply(calculate_account_age)
    df['tw_age'] = df['twitter_handle_created_date'].apply(calculate_account_age)
    df['insta_age'] = df['insta_handle_created_date'].apply(calculate_account_age)
    
    # Social media presence
    df['has_facebook'] = df['facebook_handle'].notnull().astype(int)
    df['has_twitter'] = df['twitter_handle'].notnull().astype(int)
    df['has_instagram'] = df['insta_handle'].notnull().astype(int)
    
    # Sentiment analysis on public posts
    df['public_post_sentiment'] = df['public_posts'].apply(analyze_sentiment)
    
    return df[['fb_age', 'tw_age', 'insta_age', 'has_facebook', 'has_twitter', 'has_instagram', 'public_post_sentiment']]

# Define Ingenuity Score as a weighted combination of social media age and sentiment
def calculate_ingenuity_score(row):
    # Weights for social media age and sentiment
    social_age_weight = 0.6
    sentiment_weight = 0.4

    # Social media age score is the average of all non-zero social media account ages
    social_age_score = np.mean([row['fb_age'], row['tw_age'], row['insta_age']])
    
    # Sentiment score
    sentiment_score = row['public_post_sentiment']

    # Calculate final ingenuity score
    ingenuity_score = (social_age_weight * social_age_score) + (sentiment_weight * sentiment_score * 100)
    return ingenuity_score


df = pd.read_csv('data/data.csv')


# Assume your CSV has the following columns: email, facebook_handle_created_date, twitter_handle_created_date, insta_handle_created_date, public_posts
# Sample CSV format:
# email,facebook_handle,facebook_handle_created_date,twitter_handle,twitter_handle_created_date,insta_handle,insta_handle_created_date,public_posts
# user1@example.com,@fb_user1,2010-05-01,@tw_user1,2011-03-10,@insta_user1,2012-08-15,"Great day for a hike! Loving nature."
df = feature_engineering_with_nltk(df)
print(df)

#print(df['twitter_handle_created_date'])

# Calculate ingenuity_score for each row
df['ingenuity_score'] = df.apply(calculate_ingenuity_score, axis=1)
print(df)

# Prepare Data for Model Training
X = df[['fb_age', 'tw_age', 'insta_age', 'has_facebook', 'has_twitter', 'has_instagram', 'public_post_sentiment']]
y = df['ingenuity_score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestRegressor Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


# Make Predictions and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict Scores for New Data
new_data = pd.DataFrame({
    'fb_age': [7], 
    'tw_age': [6], 
    'insta_age': [5], 
    'has_facebook': [1], 
    'has_twitter': [1], 
    'has_instagram': [1], 
    'public_post_sentiment': [0.75]  # Example sentiment score from a new public post
})

predicted_score = model.predict(new_data)
print(f"Predicted Ingenuity Score: {predicted_score[0]}")