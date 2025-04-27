import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Loading of datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Merge datasets one by one
df = pd.merge(ratings, movies, on='movieId')

# Map userId and movieId to numbers
df['userId'] = df['userId'].astype("category").cat.codes
df['movieId'] = df['movieId'].astype("category").cat.codes

# create genre features
df['genres'] = df['genres'].apply(lambda x: x.split('|')[0])  # Take only the first genre
df['genres'] = df['genres'].astype("category").cat.codes

# Select features
features = ['userId', 'movieId', 'genres']
X = df[features]
y = df['rating']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))

# Trying a custom prediction
sample = pd.DataFrame([[100, 50, 2]], columns=features)  # Example input
print("Predicted Rating:", round(model.predict(sample)[0], 2))
# visualization of ratings
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

