"""Hit Song Predictor.

This app allows a user to adjust audio features manually, or fill them from a 
Spotify track ID, to estimate the probability that a song is a hit.
"""

from __future__ import annotations
import ast
import logging
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Set up logging to track errors behind the scenes
LOGGER = logging.getLogger("hit_song_predictor")

# Define file paths relative to the current script location
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spotify_tracks.csv"
MODEL_PATH = ROOT / "hit_predictor_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
FEATURES_PATH = ROOT / "feature_names.pkl"
ARTIFACTS = (MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATA_PATH)

HIT_THRESHOLD = 0.5
_TRACK_ID_RE = re.compile(r"(?:spotify:track:|/track/)([A-Za-z0-9]+)")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Defines the rules (min, max, step) for each input feature."""
    label: str
    minimum: float
    maximum: float
    step: float
    default: float
    decimals: int
    help: str
    integer: bool = False


# Dictionary storing the rules for all 10 audio features
FEATURE_SPECS: dict[str, FeatureSpec] = {
    "danceability": FeatureSpec("Danceability", 0.0, 1.0, 0.01, 0.5, 2, "How suitable the track is for dancing, from 0 to 1."),
    "energy": FeatureSpec("Energy", 0.0, 1.0, 0.01, 0.5, 2, "Perceived intensity and activity, from 0 to 1."),
    "loudness": FeatureSpec("Loudness (dB)", -60.0, 0.0, 0.1, -10.0, 1, "Overall loudness in decibels, typically -60 to 0."),
    "speechiness": FeatureSpec("Speechiness", 0.0, 1.0, 0.01, 0.5, 2, "Presence of spoken words, from 0 to 1."),
    "acousticness": FeatureSpec("Acousticness", 0.0, 1.0, 0.01, 0.5, 2, "Confidence that the track is acoustic, from 0 to 1."),
    "instrumentalness": FeatureSpec("Instrumentalness", 0.0, 1.0, 0.01, 0.5, 2, "Confidence that the track has no vocals, from 0 to 1."),
    "liveness": FeatureSpec("Liveness", 0.0, 1.0, 0.01, 0.5, 2, "Likelihood of a live audience, from 0 to 1."),
    "valence": FeatureSpec("Valence", 0.0, 1.0, 0.01, 0.5, 2, "Musical positiveness, from 0 (somber) to 1 (cheerful)."),
    "tempo": FeatureSpec("Tempo (BPM)", 40.0, 220.0, 1.0, 120.0, 0, "Estimated tempo in beats per minute."),
    "duration_ms": FeatureSpec("Duration (ms)", 30_000, 600_000, 1_000, 200_000, 0, "Track length in milliseconds.", integer=True),
}


def feature_spec(feature: str) -> FeatureSpec:
    """Returns the spec for a feature, or a default one if not found."""
    spec = FEATURE_SPECS.get(feature)
    if spec is not None:
        return spec
    return FeatureSpec(label=feature.replace("_", " ").capitalize(), minimum=0.0, maximum=1.0, step=0.01, default=0.5, decimals=2, help="")


def clamp(value: float, spec: FeatureSpec) -> int | float:
    """Ensures a value is within the min/max range and snapped to the step size.
    
    This prevents Streamlit from crashing if a real Spotify track has a value 
    outside our allowed range (e.g., a 10-minute song when max is 600,000ms).
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{spec.label} must be finite, got {value!r}.")

    minimum = Decimal(str(spec.minimum))
    maximum = Decimal(str(spec.maximum))
    quantum = Decimal(str(spec.step))
    
    number = min(max(Decimal(str(numeric)), minimum), maximum)
    steps = ((number - minimum) / quantum).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    snapped = minimum + steps * quantum
    
    if snapped > maximum: snapped -= quantum
    if snapped < minimum: snapped = minimum

    if spec.integer: return int(snapped)
    return float(snapped)


def normalize_track_id(raw: str) -> str:
    """Extracts the track ID from a raw string, URI, or full Spotify URL."""
    text = raw.strip()
    match = _TRACK_ID_RE.search(text)
    if match:
        return match.group(1)
    return text.split("?", 1)[0].strip()


def _is_missing(value: object) -> bool:
    """Checks if a value is missing (NaN)."""
    if value is None or isinstance(value, (str, bytes, list, tuple, dict)):
        return value is None
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def display_text(value: object, fallback: str) -> str:
    """Cleans up strings for display in the UI."""
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) or fallback
    if _is_missing(value):
        return fallback

    text = str(value).strip()
    return text or fallback


def _parse_feature_names(raw: Any) -> list[str]:
    """Ensures the feature names loaded from the pkl file are valid."""
    if isinstance(raw, str) or raw is None:
        raise ValueError("feature_names.pkl must be a sequence of names, not a single string.")
    try:
        names = [str(name).strip() for name in raw]
    except TypeError as exc:
        raise ValueError("feature_names.pkl must be a sequence of names.") from exc
    names = list(dict.fromkeys(name for name in names if name))
    if not names:
        raise ValueError("feature_names.pkl did not contain any feature names.")
    return names


@st.cache_resource(show_spinner="Loading model…")
def load_model_bundle(model_mtime: int, scaler_mtime: int, features_mtime: int) -> tuple[Any, Any, list[str]]:
    """Loads the model, scaler, and features.
    
    The @st.cache_resource decorator tells Streamlit to keep this in memory.
    This makes the app MUCH faster because it doesn't reload the pkl files 
    every time you move a slider.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = _parse_feature_names(joblib.load(FEATURES_PATH))
    return model, scaler, feature_names


@st.cache_data(show_spinner="Loading tracks…")
def load_tracks(csv_mtime: int, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Loads the CSV file into memory and caches it for speed."""
    required = list(dict.fromkeys(["track_id", "track_name", "artists", *feature_names]))
    tracks = pd.read_csv(DATA_PATH, usecols=required, dtype={"track_id": "string"})
    tracks = tracks.dropna(subset=["track_id"])
    tracks["track_id"] = tracks["track_id"].astype(str).str.strip()
    tracks = tracks[tracks["track_id"] != ""]
    return tracks.set_index("track_id", drop=False)


def predict_hit(model: Any, scaler: Any, user_input: dict[str, int | float], feature_names: list[str]) -> float:
    """Feeds the inputs into the model and returns the probability (0.0 to 1.0)."""
    # 1. Format the inputs in the exact order the model expects
    row = [[float(user_input[name]) for name in feature_names]]
    matrix = pd.DataFrame(row, columns=feature_names).to_numpy(dtype=float)
    
    # 2. Scale the inputs (this is required because we scaled the training data)
    scaled = scaler.transform(matrix)
    
    # 3. Predict the probability (returns array like [prob_non_hit, prob_hit])
    probabilities = model.predict_proba(scaled)[0]
    
    # We only want the probability of being a hit (index 1)
    return float(probabilities[1])


def feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Extracts the global feature importances from the Random Forest model."""
    values = getattr(model, "feature_importances_", None)
    if values is None: return None
    
    importances = [float(value) for value in values]
    chart = pd.DataFrame({
        "feature": [feature_spec(name).label for name in feature_names],
        "importance": importances,
    })
    return chart.sort_values("importance", ascending=True)


def _sync_slider(feature: str) -> None:
    """Updates the number box when the slider moves."""
    st.session_state[f"num_{feature}"] = st.session_state[f"slider_{feature}"]


def _sync_number(feature: str) -> None:
    """Updates the slider when the number box is typed in."""
    st.session_state[f"slider_{feature}"] = st.session_state[f"num_{feature}"]


def _write_control(key: str, value: int | float) -> None:
    """Safely writes a value into Streamlit's session state."""
    current = st.session_state.get(key, _MISSING)
    if current is _MISSING or type(current) is not type(value) or current != value:
        st.session_state[key] = value


def _prepare_feature_state(feature: str) -> FeatureSpec:
    """Ensures the slider and number input always have valid, synced values."""
    spec = feature_spec(feature)
    slider_key = f"slider_{feature}"
    number_key = f"num_{feature}"
    
    if slider_key not in st.session_state:
        _write_control(slider_key, clamp(spec.default, spec))
    else:
        _write_control(slider_key, clamp(st.session_state[slider_key], spec))

    if number_key not in st.session_state:
        _write_control(number_key, st.session_state[slider_key])
    else:
        _write_control(number_key, clamp(st.session_state[number_key], spec))

    if st.session_state[slider_key] != st.session_state[number_key]:
        _write_control(slider_key, st.session_state[number_key])
    return spec


def _render_feature_input(feature: str) -> None:
    """Draws the slider and the number input side-by-side."""
    spec = _prepare_feature_state(feature)
    
    # Formatting depends on whether it's an integer or float
    minimum, maximum, step = (int(spec.minimum), int(spec.maximum), int(spec.step)) if spec.integer else (float(spec.minimum), float(spec.maximum), float(spec.step))
    number_format = "%d" if spec.integer else f"%.{spec.decimals}f"
    
    # Create two columns: one wide for slider, one narrow for number
    slider_col, number_col = st.sidebar.columns([3, 1])

    with slider_col:
        st.slider(label=spec.label, min_value=minimum, max_value=maximum, step=step, format=number_format, key=f"slider_{feature}", on_change=_sync_slider, args=(feature,), help=spec.help or None)
    with number_col:
        st.number_input(label=f"{spec.label} value", min_value=minimum, max_value=maximum, step=step, format=number_format, key=f"num_{feature}", on_change=_sync_number, args=(feature,), label_visibility="hidden")


def apply_track(track: pd.Series, feature_names: list[str]) -> list[str]:
    """Applies the data from a Spotify track to the sliders."""
    clipped: list[str] = []
    for feature in feature_names:
        spec = feature_spec(feature)
        raw = track[feature]
        
        # If data is missing or out of bounds, clamp it safely
        if _is_missing(raw):
            value = clamp(spec.default, spec)
            clipped.append(feature)
        else:
            numeric = float(raw)
            value = clamp(numeric, spec)
            if numeric < spec.minimum or numeric > spec.maximum:
                clipped.append(feature)
                
        _write_control(f"slider_{feature}", value)
        _write_control(f"num_{feature}", value)
    return clipped


def _load_selected_track(tracks: pd.DataFrame, feature_names: list[str], raw_track_id: str) -> None:
    """Finds a track in the CSV, applies its data, and refreshes the app."""
    track_id = normalize_track_id(raw_track_id)
    if not track_id:
        st.sidebar.error("Enter a Spotify track ID or track URL.")
        return
    if track_id not in tracks.index:
        st.sidebar.error("Track ID not found.")
        return

    track = tracks.loc[track_id]
    if isinstance(track, pd.DataFrame): track = track.iloc[0]

    clipped = apply_track(track, feature_names)

    name = display_text(track.get("track_name"), "Unknown title")
    artists = display_text(track.get("artists"), "Unknown artist")
    message = f"Loaded: {name} by {artists}"
    if clipped: message += f" ({len(clipped)} value(s) clipped to fit)"
    
    st.session_state["loaded_msg"] = message
    if "prediction" in st.session_state: del st.session_state["prediction"]
    st.rerun()


def render_sidebar(tracks: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, int | float], bool]:
    """Builds the entire sidebar."""
    st.sidebar.header("Song Features")

    # Show success message if a track was loaded
    if "loaded_msg" in st.session_state:
        st.sidebar.success(st.session_state["loaded_msg"])
        del st.session_state["loaded_msg"]

    # Search box
    raw_track_id = st.sidebar.text_input("Enter Spotify Track ID", placeholder="Track ID or open.spotify.com URL")
    if st.sidebar.button("Load Track Data", use_container_width=True):
        _load_selected_track(tracks, feature_names, raw_track_id)

    # Draw sliders
    for feature in feature_names:
        _render_feature_input(feature)

    # Gather all inputs
    user_input = {feature: clamp(st.session_state[f"num_{feature}"], feature_spec(feature)) for feature in feature_names}
    
    predict_clicked = st.sidebar.button("🎯 Predict Hit Potential", type="primary", use_container_width=True)
    return user_input, predict_clicked


def render_importance(model: Any, feature_names: list[str]) -> None:
    """Draws the bar chart for feature importance."""
    chart = feature_importance(model, feature_names)
    if chart is None: return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.barh(chart["feature"], chart["importance"], color="#3D5A80")
        ax.set_xlabel("Importance")
        ax.set_title("Global feature importance")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)  # Prevents memory leaks


def render_results(model: Any, scaler: Any, user_input: dict[str, int | float], feature_names: list[str], predict_clicked: bool) -> None:
    """Shows the result on the main screen."""
    # Save the current state so the score doesn't disappear when you move a slider
    signature = tuple((name, user_input[name]) for name in feature_names)

    if predict_clicked:
        probability = predict_hit(model, scaler, user_input, feature_names)
        st.session_state["prediction"] = {"signature": signature, "probability": probability}

    prediction = st.session_state.get("prediction")
    if not isinstance(prediction, dict) or prediction.get("signature") != signature:
        st.info("Set the features in the sidebar, then predict hit potential.")
        return

    probability = float(prediction["probability"])
    
    # Display the result
    score_col, verdict_col = st.columns(2)
    with score_col:
        st.metric("Hit Probability", f"{probability * 100:.1f}%")
    with verdict_col:
        if probability >= HIT_THRESHOLD:
            st.success("Likely a Hit! 🎉")
        else:
            st.warning("Less likely to be a Hit")

    st.progress(min(1.0, max(0.0, probability)))
    
    st.subheader("Feature Importance")
    st.caption("How much each feature matters to the model overall — not an explanation of this specific score.")
    render_importance(model, feature_names)


def main() -> None:
    """The main function that runs the app."""
    st.set_page_config(page_title="Hit Song Predictor", page_icon="🎵")
    st.title("🎵 Hit Song Predictor")
    st.markdown("Adjust the sliders, type exact numbers, or paste a Spotify track ID to auto-fill them.")

    # Check if all files exist before loading
    missing = [path for path in ARTIFACTS if not path.is_file()]
    if missing:
        st.error(f"Missing required file(s): {', '.join(str(p.name) for p in missing)}")
        st.stop()

    # Load model and data
    model, scaler, feature_names = load_model_bundle(MODEL_PATH.stat().st_mtime_ns, SCALER_PATH.stat().st_mtime_ns, FEATURES_PATH.stat().st_mtime_ns)
    tracks = load_tracks(DATA_PATH.stat().st_mtime_ns, tuple(feature_names))

    # Run the app
    user_input, predict_clicked = render_sidebar(tracks, feature_names)
    render_results(model, scaler, user_input, feature_names, predict_clicked)


if __name__ == "__main__":
    main()