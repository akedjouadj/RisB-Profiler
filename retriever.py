import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Use st.cache_data to cache data loading
@st.cache_data
def load_data():
    data = pd.read_csv("app_data/df_raw_counts_players_matches.csv")
    X = np.load("app_data/players_embeddings_1l128d2h_wo_teams_2454games.npy")
    with open("app_data/label2players_name_5106p.pickle", "rb") as f:
        label2players_name = pickle.load(f)
    with open("app_data/players_name2label_5106p.pickle", "rb") as f:
        players_name2label = pickle.load(f)
    
    label2players_club = {}
    label2players_positions = {}
    label2players_matches = {}
    label2players_competitions = {}
    for player_idx, player in label2players_name.items():
        player_data = data[data["player_name"] == player]
        player_clubs = list(player_data.team_name.unique())

        player_positions = list(player_data.position_name.unique())
        if "position_pad" in player_positions:
            del player_positions[player_positions.index("position_pad")]
        
        player_matches = player_data.shape[0]

        player_competitions = list(player_data.competition_name.unique()) 
        label2players_competitions[player_idx] = player_competitions
        
        label2players_club[player_idx] = player_clubs
        label2players_positions[player_idx] = player_positions
        label2players_matches[player_idx] = player_matches
    
    all_positions = sorted(data.position_name.unique())
    all_competitions = sorted(data.competition_name.unique())
    
    return data, X, label2players_name, players_name2label, label2players_club, label2players_positions, label2players_matches, label2players_competitions, all_positions, all_competitions

# Load data only once
data, X, label2players_name, players_name2label, label2players_club, label2players_positions, label2players_matches, label2players_competitions, all_positions, all_competitions = load_data()

# Function to calculate similarity
@st.cache_data
def get_similar_players(player_name, n_similar=5, excluded_positions='Goalkeeper', min_matches=10, selected_competitions=None):
    idx = players_name2label[player_name]
    player_embedding = X[idx].reshape(1, -1)
    
    similarities = cosine_similarity(player_embedding, X)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = sorted_indices[sorted_indices != idx]
    
    similar_players = []
    for player_idx in sorted_indices:
        name = label2players_name[player_idx]
        club = label2players_club[player_idx]
        position = label2players_positions[player_idx]
        matches = label2players_matches[player_idx]
        competitions = label2players_competitions[player_idx]
        
        # Apply filters
        if excluded_positions and any(pos in excluded_positions for pos in position):
            continue
        if selected_competitions and not any(comp in selected_competitions for comp in competitions):
            continue
        if matches < min_matches:
            continue


        similarity_percentage = round(100- (1 - similarities[player_idx])/2 * 100, 2)
        similar_players.append((name, club, position, matches, similarity_percentage))

        if len(similar_players) == n_similar:
            break
    
    return similar_players

# Streamlit app layout
st.sidebar.title("Football Player Retrieval")

# Sidebar inputs
player_of_interest = st.sidebar.selectbox("Select a player:", list(label2players_name.values()))
n_similar_players = st.sidebar.slider("Number of similar players:", min_value=5, max_value=50, value=5)
excluded_positions = st.sidebar.multiselect("Exclude positions:", all_positions, default = ['Goalkeeper'])
min_matches = st.sidebar.slider("Minimum number of matches played:", min_value=1, max_value=20, value=1)
selected_competitions = st.sidebar.multiselect("Select competitions:", all_competitions)  # New filter for competitions
validate = st.sidebar.button("Validate")

# Main app title
st.title("RisB-Profiler, Find the Player You Really need!")
st.write("**For clubs, scouts, analysts, and fans !**")

# Store the state of the app
if 'run_search' not in st.session_state:
    st.session_state.run_search = False

# If the user clicks validate, set the state to run the search
if validate:
    st.session_state.run_search = True

# If the state is set to run the search, display the results
if st.session_state.run_search:
    st.subheader(f"Chosen Player: {player_of_interest}")
    player_idx = players_name2label[player_of_interest]
    st.write(f"**Club(s):** {', '.join(label2players_club[player_idx])}")
    st.write(f"**Position(s):** {', '.join(label2players_positions[player_idx])}")
    st.write(f"**Total (dataset) Matches:** {label2players_matches[player_idx]}")
    
    # Get similar players
    similar_players = get_similar_players(player_of_interest, n_similar_players, excluded_positions, min_matches, selected_competitions)
    
    st.subheader(f"Top {len(similar_players)} Similar Players:")
    
    # Display similar players with their attributes
    for player, club, position, matches, similarity in similar_players:
        st.markdown(f"""
        - **Name**: {player}
        - **Club(s)**: {', '.join(club)}
        - **Position(s)**: {', '.join(position)}
        - **Matches played**: {matches}
        - **Similarity %**: {similarity} 
        """)

    # Add a button to reset the search
    if st.button("Reset Search"):
        st.session_state.run_search = False
        st.experimental_rerun()

# survery to share impressions
survey_url = "https://docs.google.com/forms/d/e/1FAIpQLSdIo7cXsc88jjwagytWk3Xb6drDsOyu7AYXYKvx5pZVNuON8Q/viewform?usp=pp_url"

st.markdown(
    f"""
    <div style='text-align: right; position: fixed; bottom: 10px; right: 10px;'>
        <a href="{survey_url}" target="_blank" style='color: blue; text-decoration: none; font-size: 16px;'>
        🚀 Share your impressions here!
        </a>
    </div>
    """,
    unsafe_allow_html=True
)