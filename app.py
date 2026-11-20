"""
CineMatch - Movie Recommendation System (Content-Based Filtering)
Tech: Streamlit, Pandas, Scikit-learn (TF-IDF + Cosine Similarity)
Run:  streamlit run app.py
"""

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# BUILT-IN DATASET (40 movies)
# Columns: title, genres, rating, year, description, poster
# poster = TMDB image path (optional). If empty or broken, a placeholder shows.
# ---------------------------------------------------------------------------
TMDB = "https://image.tmdb.org/t/p/w500"

MOVIES = [
    ("Inception", "Action, Science Fiction, Thriller", 8.8, 2010,
     "A skilled thief who steals secrets from dreams is offered a chance to erase his past by planting an idea inside a target's mind.",
     "/ljsZTbVsrQSqZgWeep2B1QiDKuh.jpg"),
    ("The Dark Knight", "Action, Drama, Thriller", 9.0, 2008,
     "Batman faces a chaotic criminal mastermind who pushes Gotham City to the edge and tests the hero's moral limits.",
     "/qJ2tW6WMUDux911r6m7haRef0WH.jpg"),
    ("Interstellar", "Science Fiction, Adventure, Drama", 8.7, 2014,
     "Astronauts travel through a wormhole near Saturn to find a new home for humanity as Earth slowly becomes unlivable.",
     "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg"),
    ("The Matrix", "Action, Science Fiction", 8.7, 1999,
     "A computer hacker discovers that reality is a simulation controlled by machines and joins a rebellion to free humanity.",
     "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg"),
    ("Avatar", "Science Fiction, Adventure, Action", 7.9, 2009,
     "A paralysed marine travels to the alien world of Pandora and must choose between following orders and protecting its people.",
     "/kyeqWdyUXW608qlYkRqosgbbJyK.jpg"),
    ("Mad Max: Fury Road", "Action, Adventure, Science Fiction", 8.1, 2015,
     "In a desert wasteland, a drifter and a rebel warrior race across the sands in a high-speed escape from a tyrant.",
     ""),
    ("Gladiator", "Action, Drama, Adventure", 8.5, 2000,
     "A betrayed Roman general becomes a gladiator and fights his way through the arena to take revenge on a corrupt emperor.",
     "/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"),
    ("John Wick", "Action, Thriller", 7.4, 2014,
     "A retired hitman returns to the criminal underworld to hunt down the gangsters who wronged him and his beloved dog.",
     ""),
    ("Raiders of the Lost Ark", "Adventure, Action", 8.4, 1981,
     "Archaeologist Indiana Jones races against villains to find a powerful ancient relic before it falls into the wrong hands.",
     ""),
    ("Jurassic Park", "Adventure, Science Fiction, Thriller", 8.2, 1993,
     "A theme park of cloned dinosaurs turns deadly when the security systems fail and visitors become prey.",
     "/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg"),
    ("Avengers: Endgame", "Action, Adventure, Science Fiction", 8.4, 2019,
     "The surviving superheroes unite for a final mission across time to undo the damage caused by a mad titan.",
     ""),
    ("Edge of Tomorrow", "Action, Science Fiction", 7.9, 2014,
     "A soldier trapped in a time loop relives the same battle against alien invaders again and again, getting better each time.",
     ""),
    ("The Shawshank Redemption", "Drama", 9.3, 1994,
     "A banker wrongly convicted of murder forms a deep friendship in prison and never gives up hope of freedom.",
     "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg"),
    ("Forrest Gump", "Drama, Romance, Comedy", 8.8, 1994,
     "A kind-hearted man with a low IQ unknowingly witnesses and influences decades of American history while pining for his childhood love.",
     "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg"),
    ("Pulp Fiction", "Thriller, Drama", 8.9, 1994,
     "The lives of hitmen, a boxer and a gangster's wife intertwine in a series of darkly funny stories of crime in Los Angeles.",
     "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg"),
    ("Fight Club", "Drama, Thriller", 8.8, 1999,
     "An insomniac office worker and a reckless soap salesman start an underground fight club that spirals into something much bigger.",
     "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"),
    ("Dangal", "Drama", 8.3, 2016,
     "A former wrestler trains his daughters to become world-class wrestlers and fights social expectations to help them win for India.",
     ""),
    ("Titanic", "Romance, Drama", 7.9, 1997,
     "A young artist and a wealthy girl fall in love aboard the doomed ocean liner on its tragic maiden voyage.",
     "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"),
    ("The Notebook", "Romance, Drama", 7.8, 2004,
     "A poor young man and a rich young woman fall in love one summer, and a lifetime of love is told through an old man's reading.",
     ""),
    ("La La Land", "Romance, Drama, Comedy", 8.0, 2016,
     "A jazz pianist and an aspiring actress fall in love in Los Angeles while chasing their dreams and facing hard choices.",
     ""),
    ("Pride and Prejudice", "Romance, Drama", 7.8, 2005,
     "Elizabeth Bennet navigates family pressure, class differences and her first impressions of the proud Mr. Darcy.",
     ""),
    ("Before Sunrise", "Romance, Drama", 8.1, 1995,
     "Two strangers meet on a train and spend one night walking through Vienna, talking about love, life and time.",
     ""),
    ("Zindagi Na Milegi Dobara", "Comedy, Drama, Adventure, Romance", 8.2, 2011,
     "Three childhood friends take a bachelor road trip across Spain, facing their fears and rediscovering what matters in life.",
     ""),
    ("Toy Story", "Animation, Comedy, Adventure", 8.3, 1995,
     "A cowboy doll feels threatened when a flashy space ranger toy becomes his owner's new favourite, and the two must work together.",
     "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"),
    ("Finding Nemo", "Animation, Adventure, Comedy", 8.2, 2003,
     "An anxious clownfish crosses the ocean with a forgetful blue tang to rescue his son from a fish tank in Sydney.",
     ""),
    ("The Lion King", "Animation, Drama, Adventure", 8.5, 1994,
     "A young lion prince flees his kingdom after his father's death, then returns as an adult to reclaim his place.",
     ""),
    ("Spirited Away", "Animation, Adventure, Drama", 8.6, 2001,
     "A girl wanders into a spirit world and must work in a magical bathhouse to free her parents and find her way home.",
     ""),
    ("Up", "Animation, Adventure, Comedy", 8.3, 2009,
     "An elderly widower ties thousands of balloons to his house and flies to South America with an eager young scout.",
     ""),
    ("Inside Out", "Animation, Comedy, Drama", 8.1, 2015,
     "The emotions inside a young girl's mind, Joy and Sadness among them, struggle to guide her through a big move to a new city.",
     ""),
    ("The Hangover", "Comedy", 7.7, 2009,
     "Three friends wake up after a wild bachelor party in Las Vegas with no memory and must find the missing groom.",
     ""),
    ("Superbad", "Comedy", 7.6, 2007,
     "Two awkward high school friends try to buy alcohol for a party before graduation, leading to a chaotic night.",
     ""),
    ("3 Idiots", "Comedy, Drama", 8.4, 2009,
     "Two friends look back on their engineering college days and the free-spirited friend who taught them to follow their passion.",
     ""),
    ("The Grand Budapest Hotel", "Comedy, Adventure, Drama", 8.1, 2014,
     "A legendary concierge and his young lobby boy are caught up in the theft of a priceless painting and a family feud.",
     ""),
    ("Se7en", "Thriller, Drama", 8.6, 1995,
     "Two detectives hunt a serial killer who uses the seven deadly sins as the pattern for his gruesome crimes.",
     ""),
    ("Gone Girl", "Thriller, Drama", 8.1, 2014,
     "When a woman disappears on her fifth wedding anniversary, her husband becomes the prime suspect as dark secrets surface.",
     ""),
    ("Shutter Island", "Thriller, Drama", 8.2, 2010,
     "A U.S. marshal investigates the disappearance of a patient from a remote hospital for the criminally insane and questions his own mind.",
     ""),
    ("The Silence of the Lambs", "Thriller, Drama", 8.6, 1991,
     "A young FBI trainee seeks help from an imprisoned genius killer to catch another serial murderer.",
     ""),
    ("Blade Runner 2049", "Science Fiction, Drama, Thriller", 8.0, 2017,
     "A young replicant hunter uncovers a long-buried secret that leads him to search for a missing former blade runner.",
     ""),
    ("The Martian", "Science Fiction, Adventure, Drama", 8.0, 2015,
     "An astronaut is stranded alone on Mars and uses science and humour to survive until a rescue can reach him.",
     ""),
    ("Kung Fu Panda", "Animation, Action, Comedy", 7.6, 2008,
     "A clumsy, food-loving panda is chosen as the legendary Dragon Warrior and must master kung fu to defeat a vengeful snow leopard.",
     ""),
]

GENRE_COLORS = {
    "Action": "#7a1010", "Adventure": "#0f5a3a", "Science Fiction": "#143a7a",
    "Comedy": "#8a6a00", "Drama": "#4a2a6a", "Thriller": "#243b55",
    "Romance": "#8a1a4a", "Animation": "#0f6a7a",
}


# ---------------------------------------------------------------------------
# DATA + ML MODEL (built only once thanks to st.cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Build the DataFrame, TF-IDF matrix and cosine similarity matrix once."""
    df = pd.DataFrame(MOVIES, columns=["title", "genres", "rating", "year", "description", "poster"])
    # Combine genres (repeated to give them more weight) and description into one text feature
    df["features"] = (df["genres"].str.replace(",", " ") + " ") * 2 + df["description"]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["features"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, similarity


def recommend(title, df, similarity, top_n=5):
    """Return the top_n most similar movies (never includes the selected movie)."""
    matches = df.index[df["title"] == title].tolist()
    if not matches:
        return pd.DataFrame()
    idx = matches[0]
    scores = [(i, s) for i, s in enumerate(similarity[idx]) if i != idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_n]
    result = df.iloc[[i for i, _ in top]].copy()
    result["match"] = [round(s * 100) for _, s in top]
    return result


df, similarity = load_model()

# ---------------------------------------------------------------------------
# SESSION STATE + NAVIGATION
# ---------------------------------------------------------------------------
PAGES = ["Home", "Explore", "Search", "Recommendations", "Watchlist", "About", "Movie Details"]

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = df.iloc[0]["title"]
if "recommend_movie" not in st.session_state:
    st.session_state.recommend_movie = df.iloc[0]["title"]
if "previous_page" not in st.session_state:
    st.session_state.previous_page = "Explore"
if "scroll_top" not in st.session_state:
    st.session_state.scroll_top = False
if "scroll_runs" not in st.session_state:
    st.session_state.scroll_runs = 0


def go_to(page):
    st.session_state.nav = page
    st.session_state.scroll_top = True  # new page should start at the top


def open_movie(title):
    # Remember where the user came from so Back returns to the same section.
    current_page = st.session_state.get("nav", "Explore")
    if current_page != "Movie Details":
        st.session_state.previous_page = current_page
    st.session_state.selected_movie = title
    st.session_state.nav = "Movie Details"
    st.session_state.scroll_top = True


def open_recommendations(title):
    st.session_state.recommend_movie = title
    st.session_state.nav = "Recommendations"
    st.session_state.scroll_top = True


def toggle_watchlist(title):
    if title in st.session_state.watchlist:
        st.session_state.watchlist.remove(title)
    else:
        st.session_state.watchlist.append(title)


# ---------------------------------------------------------------------------
# CINEMATIC CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp { background: #090909; color: #f5f5f5; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: #090909; box-shadow: none; }
    section[data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1300px; padding-top: 4rem; padding-bottom: 3rem; }
    h1, h2, h3, h4, p, label { color: #f5f5f5; }
    .logo { font-size: 1.8rem; font-weight: 800; letter-spacing: 1.5px; margin: 0; color: #E50914 !important; }
    .logo span { color: #fff !important; }
    .st-key-topbar { padding: 2px 0 6px; margin-bottom: 18px; border-bottom: 1px solid #1c1c1c; }
    .st-key-topbar .stButton button { background: transparent; border: none; border-bottom: 3px solid transparent; border-radius: 0; min-height: 2.8rem; font-weight: 600; }
    .st-key-topbar .stButton button p { color: #b3b3b3; margin: 0; }
    .st-key-topbar .stButton button:hover { background: transparent; border-bottom-color: #6b1217; }
    .st-key-topbar .stButton button:hover p { color: #fff; }
    .st-key-topbar .stButton button[kind="primary"], .st-key-topbar .stButton button[data-testid="stBaseButton-primary"] { background: transparent; border-bottom-color: #E50914; }
    .st-key-topbar .stButton button[kind="primary"] p, .st-key-topbar .stButton button[data-testid="stBaseButton-primary"] p { color: #fff; }
    .st-key-topbar .stButton button[kind="primary"]:hover, .st-key-topbar .stButton button[data-testid="stBaseButton-primary"]:hover { background: transparent; border-bottom-color: #ff2a35; }
    .tagline { color: #a6a6a6 !important; margin: -4px 0 15px 2px; font-size: .9rem; }
    .section-title { font-size: 1.45rem; font-weight: 800; margin: 28px 0 14px; border-left: 4px solid #E50914; padding-left: 12px; }
    .hero { display:flex; gap:28px; align-items:center; padding:30px; min-height:270px; border-radius:18px; margin:14px 0 20px; background:linear-gradient(105deg,#3b080b 0%,#1a1112 48%,#151515 100%); }
    .hero-copy h1 { font-size:2.6rem; margin:8px 0; font-weight:800; }
    .hero-copy p { color:#c6c6c6; line-height:1.65; max-width:760px; }
    .hero-poster { width:130px; min-width:130px; aspect-ratio:2/3; position:relative; overflow:hidden; border-radius:12px; box-shadow:0 14px 35px #0009; }
    .hero-poster img,.poster img,.detail-poster img { width:100%; height:100%; object-fit:cover; position:absolute; inset:0; }
    .poster { width:100%; aspect-ratio:2/3; position:relative; overflow:hidden; background:#202020; }
    .poster .ph, .hero-poster .ph, .detail-poster .ph { position:absolute; inset:0; display:flex; flex-direction:column; gap:8px; align-items:center; justify-content:center; text-align:center; padding:12px; font-weight:700; font-size:.95rem; }
    .card { background:#141414; border-radius:10px; overflow:hidden; transition:transform .2s,border-color .2s,box-shadow .2s; margin-bottom:8px; }
    .card:hover { transform:translateY(-4px); box-shadow:0 12px 28px #e5091433; }
    .card-body { padding:11px 12px 13px; }
    .card-title { font-size:.96rem; font-weight:700; min-height:2.4em; }
    .card-genre { color:#ef555c !important; font-size:.74rem; margin:5px 0; min-height:2.2em; }
    .card-rating { color:#f5c518 !important; font-size:.85rem; font-weight:700; }
    .ph-icon { font-size:1.4rem; }
    .ph-title { font-size:1.02rem; font-weight:800; line-height:1.25; }
    .ph-meta { font-size:.68rem; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; color:#ffffffaa; }
    [data-testid="stAppDeployButton"], .stDeployButton { display:none; }
    .muted { color:#a9a9a9 !important; }
    .match { float:right; background:#145b3c; color:white !important; padding:2px 8px; border-radius:99px; font-size:.7rem; }
    .stButton button { background:#1f1f1f; color:#fff; border:1px solid #3a3a3a; border-radius:8px; font-weight:650; min-height:2.55rem; width:100%; }
    .stButton button:hover { background:#262626; border-color:#E50914; color:#fff; }
    .stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] { background:#E50914; border-color:#E50914; }
    .stButton button[kind="primary"]:hover, .stButton button[data-testid="stBaseButton-primary"]:hover { background:#b20710; border-color:#b20710; }
    .stTextInput input, div[data-baseweb="select"] > div { background:#191919 !important; color:#fff !important; border:1px solid #333 !important; border-radius:8px; }
    .info-box,.stat-box { background:#171717; border-radius:12px; padding:15px 18px; margin:8px 0; }
    .detail-poster { width:100%; max-width:310px; aspect-ratio:2/3; position:relative; overflow:hidden; border-radius:14px; box-shadow:0 16px 40px #0009; }
    .detail-title { font-size:2.4rem; font-weight:800; margin:0 0 8px; }
    .eyebrow { color:#ef555c !important; font-size:.78rem; font-weight:800; letter-spacing:1.5px; }
    .stCaption { color:#aaa !important; }
    @media (max-width: 700px) { .hero { flex-direction:column; align-items:flex-start; padding:20px; } .hero-copy h1 {font-size:2rem;} .detail-title {font-size:1.8rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def poster_html(row, css_class="poster"):
    """Poster with a genre-coloured fallback when an image is unavailable."""
    primary = row["genres"].split(",")[0].strip()
    color = GENRE_COLORS.get(primary, "#333333")
    title = html.escape(str(row["title"]))
    img = (
        f'<img src="{TMDB}{html.escape(str(row["poster"]), quote=True)}" '
        f'alt="{title} poster" onerror="this.style.display=\'none\'">'
        if row["poster"] else ""
    )
    return (
        f'<div class="{css_class}" style="background:linear-gradient(155deg,{color},#101010);">'
        f'<div class="ph"><span class="ph-icon">🎬</span><span class="ph-title">{title}</span>'
        f'<span class="ph-meta">{row["year"]} · {html.escape(primary)}</span></div>{img}</div>'
    )


def card_html(row, match=None):
    badge = f'<span class="match">{match}% match</span>' if match is not None else ""
    return (
        f'<div class="card">{poster_html(row)}<div class="card-body">{badge}'
        f'<div class="card-title">{html.escape(str(row["title"]))}</div>'
        f'<div class="card-genre">{html.escape(str(row["genres"]))}</div>'
        f'<div class="card-rating">⭐ {row["rating"]} <span class="muted">· {row["year"]}</span></div>'
        f'</div></div>'
    )


def show_grid(movies, key_prefix, per_row=6, show_match=False):
    """Show movie cards with working details and watchlist actions."""
    rows = list(movies.iterrows())
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, (idx, row) in zip(cols, rows[start:start + per_row]):
            with col:
                match = row["match"] if show_match and "match" in row else None
                st.markdown(card_html(row, match), unsafe_allow_html=True)
                st.button(
                    "View Details",
                    key=f"{key_prefix}_details_{idx}",
                    type="primary",
                    on_click=open_movie,
                    args=(row["title"],),
                )
                saved = row["title"] in st.session_state.watchlist
                st.button(
                    "✓ In Watchlist" if saved else "＋ Watchlist",
                    key=f"{key_prefix}_watch_{idx}",
                    on_click=toggle_watchlist,
                    args=(row["title"],),
                )


# ---------------------------------------------------------------------------
# HEADER + NAVIGATION
# ---------------------------------------------------------------------------
# One top bar: logo on the left, page links next to it. The current page is underlined in red.
# "Movie Details" only appears in the bar while you are on that page.
nav_items = [p for p in PAGES if p != "Movie Details" or st.session_state.nav == "Movie Details"]
weights = [1.6] + [1.5 if p == "Recommendations" else 1.2 if p == "Movie Details" else 1 for p in nav_items] + [3]

try:
    topbar = st.container(key="topbar")  # the key lets the CSS style only this bar
except TypeError:  # very old Streamlit: still works, just less polished
    topbar = st.container()

with topbar:
    try:
        cols = st.columns(weights, vertical_alignment="center")
    except TypeError:
        cols = st.columns(weights)
    cols[0].markdown('<p class="logo">CINE<span>MATCH</span></p>', unsafe_allow_html=True)
    for col, name in zip(cols[1:-1], nav_items):
        with col:
            st.button(
                name,
                key=f"nav_{name}",
                on_click=go_to,
                args=(name,),
                type="primary" if st.session_state.nav == name else "secondary",
            )
page = st.session_state.nav

# After opening a new page, scroll the browser back to the top.
if st.session_state.scroll_top:
    st.session_state.scroll_top = False
    st.session_state.scroll_runs += 1
    components.html(
        f"""<script>
        // run {st.session_state.scroll_runs}
        function toTop() {{
            const d = window.parent.document;
            ['[data-testid="stMain"]', 'section.main', '.main', '[data-testid="stAppViewContainer"]'].forEach(function (s) {{
                const el = d.querySelector(s);
                if (el) {{ el.scrollTop = 0; el.scrollTo(0, 0); }}
            }});
            window.parent.scrollTo(0, 0);
        }}
        toTop();
        setTimeout(toTop, 150);
        setTimeout(toTop, 500);
        </script>""",
        height=0,
    )

if page == "Home":
    st.markdown(
        f'<div class="muted" style="margin:0 0 6px;">🎞 {len(df)} movies · '
        f'{len({g.strip() for genres in df["genres"] for g in genres.split(",")})} genres · '
        f'Content-based recommendations</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
if page == "Home":
    featured = df[df["title"] == "Inception"].iloc[0]
    st.markdown(
        f'<div class="hero">{poster_html(featured, "hero-poster")}<div class="hero-copy">'
        f'<span class="eyebrow">CINEMATCH FEATURED PICK</span>'
        f'<h1>{html.escape(featured["title"])}</h1>'
        f'<p>⭐ {featured["rating"]} &nbsp; · &nbsp; {featured["year"]} &nbsp; · &nbsp; {html.escape(featured["genres"])}</p>'
        f'<p>{html.escape(featured["description"])}</p></div></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns([1, 1, 3])
    a.button("Explore Movies", on_click=go_to, args=("Explore",), key="home_explore", type="primary")
    b.button("Get Recommendations", on_click=go_to, args=("Recommendations",), key="home_recs", type="primary")
    st.markdown('<div class="section-title">Popular Movies</div>', unsafe_allow_html=True)
    show_grid(df.sort_values("rating", ascending=False).head(10), "home")

elif page == "Explore":
    st.markdown('<div class="section-title">Explore the Movie Library</div>', unsafe_allow_html=True)
    all_genres = sorted({g.strip() for gs in df["genres"] for g in gs.split(",")})
    chosen = st.selectbox("Filter by genre", ["All genres"] + all_genres)
    result = df if chosen == "All genres" else df[df["genres"].str.contains(chosen, regex=False)]
    st.caption(f"Showing {len(result)} of {len(df)} movies")
    show_grid(result, "explore")

elif page == "Search":
    st.markdown('<div class="section-title">Find a Movie</div>', unsafe_allow_html=True)
    query = st.text_input("Search by title", placeholder="Try Inception, Avatar or 3 Idiots...").strip()
    if not query:
        st.info("Enter a movie title to search the CineMatch library.")
    else:
        found = df[df["title"].str.contains(query, case=False, regex=False)]
        if found.empty:
            st.warning(f"No movies found for '{query}'. Try another title.")
        else:
            st.caption(f"{len(found)} result(s)")
            show_grid(found, "search")

elif page == "Recommendations":
    st.markdown('<div class="section-title">Movie Recommendations</div>', unsafe_allow_html=True)
    st.write("Choose a movie. CineMatch compares its genres and description with the other movies using TF-IDF and cosine similarity.")
    titles = df["title"].tolist()
    default_title = st.session_state.recommend_movie
    default_index = titles.index(default_title) if default_title in titles else 0
    selected = st.selectbox("Select a movie you like", titles, index=default_index, key="recommendation_movie_select")
    if selected:
        st.session_state.recommend_movie = selected  # remember the choice
        selected_row = df[df["title"] == selected].iloc[0]
        st.markdown(f'<div class="info-box"><b>Your selection:</b> {html.escape(selected)} &nbsp; · &nbsp; ⭐ {selected_row["rating"]} &nbsp; · &nbsp; {html.escape(selected_row["genres"])}</div>', unsafe_allow_html=True)
        recs = recommend(selected, df, similarity, top_n=5)
        st.markdown('<div class="section-title">Top 5 Similar Movies</div>', unsafe_allow_html=True)
        show_grid(recs, "recommendations", per_row=6, show_match=True)

elif page == "Watchlist":
    st.markdown('<div class="section-title">My Watchlist</div>', unsafe_allow_html=True)
    saved = df[df["title"].isin(st.session_state.watchlist)]
    if saved.empty:
        st.info("Your watchlist is empty. Add movies using the Watchlist button on any movie card.")
    else:
        st.caption(f"{len(saved)} movie(s) saved")
        show_grid(saved, "watchlist")

elif page == "About":
    st.markdown('<div class="section-title">About CineMatch</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box"><span class="eyebrow">PROJECT OVERVIEW</span><p>CineMatch is a content-based movie recommendation mini-project. It recommends movies by comparing the genres and descriptions of films in its {len(df)}-movie sample library.</p></div>', unsafe_allow_html=True)
    sections = {
        "Content-Based Filtering": "This method recommends movies with characteristics similar to a movie the user already likes. CineMatch compares genres and descriptions; it does not require other users' viewing histories.",
        "TF-IDF Vectorization": "TF-IDF converts movie text into numerical values. It gives more importance to useful words and less importance to words that appear in many descriptions.",
        "Cosine Similarity": "Cosine similarity compares the TF-IDF vectors. A higher score means the movie descriptions and genres are more similar. CineMatch returns the five highest-scoring matches, excluding the selected movie.",
        "Technologies Used": "Python, Streamlit, Pandas and Scikit-learn. The recommendation model uses TfidfVectorizer and cosine_similarity.",
    }
    for heading, body in sections.items():
        st.markdown(f'<div class="info-box"><b>{heading}</b><br><span class="muted">{body}</span></div>', unsafe_allow_html=True)

elif page == "Movie Details":
    selected_title = st.session_state.selected_movie
    selected_rows = df[df["title"] == selected_title]
    if selected_rows.empty:
        st.warning("Movie not found. Please return to Explore.")
        st.button("Back to Explore", on_click=go_to, args=("Explore",), key="details_missing_back")
    else:
        movie = selected_rows.iloc[0]
        left, right = st.columns([1, 2.1], gap="large")
        with left:
            st.markdown(poster_html(movie, "detail-poster"), unsafe_allow_html=True)
        with right:
            st.markdown('<span class="eyebrow">MOVIE DETAILS</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="detail-title">{html.escape(movie["title"])}</div>', unsafe_allow_html=True)
            st.markdown(f'⭐ **{movie["rating"]}/10** &nbsp; · &nbsp; **{movie["year"]}**', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#ef555c;font-weight:650;">{html.escape(movie["genres"])}</p>', unsafe_allow_html=True)
            st.markdown("### Overview")
            st.write(movie["description"])
            st.write("")
            saved = movie["title"] in st.session_state.watchlist
            st.button(
                "✓ Remove from Watchlist" if saved else "＋ Add to Watchlist",
                key="details_watchlist_button",
                type="primary",
                on_click=toggle_watchlist,
                args=(movie["title"],),
            )
            st.button(
                "✨ Recommend Similar Movies",
                key="details_recommend_button",
                on_click=open_recommendations,
                args=(movie["title"],),
            )
            st.button(
                "← Back",
                key="details_back_button",
                on_click=go_to,
                args=(st.session_state.previous_page,),
            )
        st.markdown('<div class="section-title">You may also like</div>', unsafe_allow_html=True)
        related = recommend(movie["title"], df, similarity, top_n=5)
        show_grid(related, "details_related", per_row=6, show_match=True)