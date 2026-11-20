# CineMatch – Movie Recommendation System Using Machine Learning

CineMatch is a Netflix-inspired web app that recommends movies using **content-based filtering**. Pick a movie you like and CineMatch finds the 5 most similar movies by comparing their genres and descriptions with TF-IDF and cosine similarity.

## Features

- **Home** – featured movie banner, short introduction and popular movies
- **Explore** – 40 built-in movies with genre filtering
- **Search** – search movies by title (friendly message when nothing matches)
- **Recommendations** – top 5 similar movies from real cosine similarity scores
- **Watchlist** – add and remove movies (stored in Streamlit `session_state`)
- **About** – explanation of the project and the machine learning concepts

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn (TfidfVectorizer, cosine_similarity)
- Custom CSS

## Installation

1. Install Python 3.9 or newer from https://www.python.org (tick "Add Python to PATH").
2. Open the `CineMatch` folder in a terminal (PowerShell).
3. (Optional) Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

4. Install the dependencies:

```powershell
pip install -r requirements.txt
```

## How to Run

```powershell
streamlit run app.py
```

The app opens at http://localhost:8501 in your browser.

## Machine Learning Explanation

1. **Dataset** – a Pandas DataFrame of 40 movies (title, genres, rating, year, description).
2. **Text feature** – the genres (repeated twice for extra weight) and the description are joined into one text.
3. **TF-IDF** – `TfidfVectorizer` converts each text into a numeric vector. Words that are frequent in one movie but rare overall get higher weights. English stop words such as "the" and "and" are removed.
4. **Cosine similarity** – `cosine_similarity` compares every pair of movie vectors and gives a score between 0 and 1.
5. **Recommendation** – for the selected movie, the scores are sorted, the movie itself is excluded, and the top 5 are shown with their match percentage.

The model is built only once using `st.cache_resource`, so the app stays fast.

## Project Limitations

- Only 40 sample movies, stored inside `app.py`.
- Recommendations use only genres and descriptions (no actors, directors or user ratings).
- The watchlist is stored in `session_state`, so it resets when the browser page is refreshed or closed.
- Some posters may not load (they need internet access); a placeholder is shown instead.
- No user login or database.
