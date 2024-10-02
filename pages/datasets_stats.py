import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Cache data loading to avoid reloading on every interaction
@st.cache_data
def load_data():
    data = pd.read_csv("app_data/df_raw_counts_players_matches.csv")
    X = np.load("app_data/players_embeddings_1l128d2h_wo_teams_2454games.npy")
    with open("app_data/label2players_name_5106p.pickle", "rb") as f:
        label2players_name = pickle.load(f)
    with open("app_data/players_name2label_5106p.pickle", "rb") as f:
        players_name2label = pickle.load(f)
    
    return data, label2players_name, players_name2label

data, label2players_name, players_name2label = load_data()

# Helper function for creating plots
def plot_barplot(data, x, y, title, xlabel, ylabel, orientation='h', figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, orient=orientation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Helper function for plotting stats
# Helper function for plotting stats
def plot_team_stats(team_data, team_name):
    st.subheader(f"Statistics for team: {team_name}")
    
    # Remove duplicates based on match_id to count each match only once
    team_data_unique_matches = team_data.drop_duplicates(subset=['match_id'])
    
    # Total matches
    total_matches = team_data_unique_matches.shape[0]
    st.write(f"**Total matches played by {team_name}:** {total_matches}")
    
    # Matches by competition (remove duplicates by match_id to avoid counting multiple players in same match)
    comp_data = team_data_unique_matches['competition_name'].value_counts().reset_index()
    comp_data.columns = ['Competition', 'Matches']
    plot_barplot(comp_data, x='Matches', y='Competition', title=f'Matches by Competition for {team_name}', xlabel='Number of Matches', ylabel='Competition')
    
    # Matches by season (remove duplicates by match_id)
    season_data = team_data_unique_matches['season_name'].value_counts().reset_index()
    season_data.columns = ['Season', 'Matches']
    plot_barplot(season_data, x='Matches', y='Season', title=f'Matches by Season for {team_name}', xlabel='Number of Matches', ylabel='Season')
    
    # Matches by players (we do not need to remove duplicates here because we want to see the total per player)
    player_data = team_data['player_name'].value_counts().reset_index()
    player_data.columns = ['Player', 'Matches']
    plot_barplot(player_data, x='Matches', y='Player', title=f'Matches by Player for {team_name}', xlabel='Number of Matches', ylabel='Player', figsize=(10, 20))


# Helper function for plotting player stats
# Helper function for plotting player stats
def plot_player_stats(player_data, player_name):
    st.subheader(f"Statistics for player: {player_name}")
    
    # Remove duplicates based on match_id to count each match only once
    player_data_unique_matches = player_data.drop_duplicates(subset=['match_id'])
    
    # Total matches
    total_matches = player_data_unique_matches.shape[0]
    st.write(f"**Total matches played by {player_name}:** {total_matches}")
    
    # Matches by team (remove duplicates by match_id)
    team_data = player_data_unique_matches['team_name'].value_counts().reset_index()
    team_data.columns = ['Team', 'Matches']
    plot_barplot(team_data, x='Matches', y='Team', title=f'Matches by Team for {player_name}', xlabel='Number of Matches', ylabel='Team')
    
    # Matches by competition (remove duplicates by match_id)
    comp_data = player_data_unique_matches['competition_name'].value_counts().reset_index()
    comp_data.columns = ['Competition', 'Matches']
    plot_barplot(comp_data, x='Matches', y='Competition', title=f'Matches by Competition for {player_name}', xlabel='Number of Matches', ylabel='Competition')
    
    # Matches by season (remove duplicates by match_id)
    season_data = player_data_unique_matches['season_name'].value_counts().reset_index()
    season_data.columns = ['Season', 'Matches']
    plot_barplot(season_data, x='Matches', y='Season', title=f'Matches by Season for {player_name}', xlabel='Number of Matches', ylabel='Season')

# Global stats page function
def global_stats_page():
    st.title("Global Dataset Statistics Used to Train the Model Behind the App")
    
    # Remove duplicates based on match_id to count each match only once
    data_unique_matches = data.drop_duplicates(subset=['match_id'])
    
    # Total matches
    total_matches = data_unique_matches.shape[0]
    total_teams = data['team_name'].nunique()
    total_players = data['player_name'].nunique()
    st.write(f"**Total number of unique matches in the dataset:** {total_matches}")
    st.write(f"**Total number of teams in the dataset:** {total_teams}")
    st.write(f"**Total number of players in the dataset:** {total_players}")
    
    # Matches by competition (remove duplicates by match_id)
    comp_data = data_unique_matches['competition_name'].value_counts().reset_index()
    comp_data.columns = ['Competition', 'Matches']
    plot_barplot(comp_data, x='Matches', y='Competition', title='Total Unique Matches by Competition', xlabel='Number of Matches', ylabel='Competition')
    
    # Matches by season (remove duplicates by match_id)
    season_data = data_unique_matches['season_name'].value_counts().reset_index()
    season_data.columns = ['Season', 'Matches']
    plot_barplot(season_data, x='Matches', y='Season', title='Total Unique Matches by Season', xlabel='Number of Matches', ylabel='Season')

# Sidebar input for team and player selection
with st.sidebar:
    st.header("Filter Options")
    
    # Player selection form
    with st.form(key='player_selection_form'):
        selected_player = st.selectbox("Select a player:", list(label2players_name.values()))
        player_submit_button = st.form_submit_button(label="Validate Player")
    
    # Team selection form
    with st.form(key='team_selection_form'):
        selected_team = st.selectbox("Select a team:", data['team_name'].unique())
        team_submit_button = st.form_submit_button(label="Validate Team")

# Display team stats if a team is selected
if team_submit_button:
    team_data = data[data['team_name'] == selected_team]
    plot_team_stats(team_data, selected_team)

# Display player stats if a player is selected
if player_submit_button:
    player_data = data[data['player_name'] == selected_player]
    plot_player_stats(player_data, selected_player)

# Display the global stats page by default
if not (team_submit_button or player_submit_button):
    global_stats_page()
