import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load('hit_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
df = pd.read_csv("data/spotify_tracks.csv")

st.set_page_config(page_title="Hit Song Predictor", page_icon="🎵")
st.title("🎵 Hit Song Predictor")
st.markdown("Adjust the sliders or paste a Spotify track ID to auto-fill them.")

st.sidebar.header("Song Features")
track_id_input = st.sidebar.text_input("Enter Spotify Track ID")

if st.sidebar.button("Load Track Data"):
    if track_id_input:
        match = df[df['track_id'] == track_id_input]
        if not match.empty:
            track = match.iloc[0]
            st.sidebar.success(f"Loaded: {track['track_name']} by {track['artists']}")
            st.session_state.loaded_values = {
                'danceability': float(track['danceability']), 'energy': float(track['energy']),
                'loudness': float(track['loudness']), 'speechiness': float(track['speechiness']),
                'acousticness': float(track['acousticness']), 'instrumentalness': float(track['instrumentalness']),
                'liveness': float(track['liveness']), 'valence': float(track['valence']),
                'tempo': float(track['tempo']), 'duration_ms': int(track['duration_ms'])
            }
        else:
            st.sidebar.error("Track ID not found.")

defaults = st.session_state.get('loaded_values', {
    'danceability': 0.5, 'energy': 0.5, 'loudness': -10.0, 'speechiness': 0.5, 
    'acousticness': 0.5, 'instrumentalness': 0.5, 'liveness': 0.5, 'valence': 0.5, 
    'tempo': 120.0, 'duration_ms': 200000
})

user_input = {}
for feat in feature_names:
    if feat == 'duration_ms':
        user_input[feat] = st.sidebar.slider(f"{feat} (ms)", 30000, 600000, int(defaults.get(feat, 200000)), step=1000)
    elif feat == 'tempo':
        user_input[feat] = st.sidebar.slider(f"{feat} (BPM)", 40.0, 220.0, float(defaults.get(feat, 120.0)), step=1.0)
    elif feat == 'loudness':
        user_input[feat] = st.sidebar.slider(f"{feat} (dB)", -60.0, 0.0, float(defaults.get(feat, -10.0)), step=0.1)
    else:
        user_input[feat] = st.sidebar.slider(feat, 0.0, 1.0, float(defaults.get(feat, 0.5)), step=0.01)

if st.sidebar.button("🎯 Predict Hit Potential"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0, 1]
    hit_prob = proba * 100
    importances = model.feature_importances_

    st.header("📊 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hit Probability", f"{hit_prob:.1f}%")
    with col2:
        if hit_prob >= 50:
            st.success("Likely a Hit! 🎉")
        else:
            st.warning("Less likely to be a Hit")

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=True)
    ax.barh(feat_imp['feature'], feat_imp['importance'])
    st.pyplot(fig)