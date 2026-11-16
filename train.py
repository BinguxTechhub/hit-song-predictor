import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading data...")
df = pd.read_csv("data/spotify_tracks.csv")
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
df = df.dropna(subset=['popularity'] + audio_features)

high = df['popularity'].quantile(0.90)
low = df['popularity'].quantile(0.20)
df = df[(df['popularity'] >= high) | (df['popularity'] <= low)].copy()
df['hit'] = (df['popularity'] >= high).astype(int)

X = df[audio_features]
y = df['hit']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

joblib.dump(model, 'hit_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(audio_features, 'feature_names.pkl')
print("Model trained and saved successfully!")