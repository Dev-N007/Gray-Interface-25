# 🎧 Spotify Song Recommender System (2023)

This is a content-based song recommender system built using the **Top Spotify Songs of 2023** dataset from Kaggle. The system recommends songs based on audio features like energy, valence, danceability, and more using **cosine similarity**.

---

## 📁 Dataset

- **Source**: [Kaggle - Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)
- **Encoding**: `cp1252` (to correctly read special characters)

---

## 🔍 Features Used

The system uses the following features for similarity calculation:

- `bpm`
- `danceability_%`
- `valence_%`
- `energy_%`
- `acousticness_%`
- `instrumentalness_%`
- `liveness_%`
- `speechiness_%`
- `key` (converted to numeric)
- `mode` (Major/Minor encoded)

---

## 🧠 Recommender System Logic

- **Approach**: Content-based filtering
- **Algorithm**: Cosine similarity on scaled numeric features
- **Scaler**: `MinMaxScaler` from scikit-learn

### 🧪 Example Usage

# Kaggle Link:
https://www.kaggle.com/code/adityamishra2505/notebook86e10f8c6f

```python
get_recommendations("Flowers")
