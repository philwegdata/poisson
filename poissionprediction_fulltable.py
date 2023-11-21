import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Function to load and preprocess data for a given league
def load_data(league):
    file_paths = {
        'Bundesliga': 'https://www.football-data.co.uk/mmz4281/2324/D1.csv',
        'La Liga': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
        '2. Bundesliga': 'https://www.football-data.co.uk/mmz4281/2324/D2.csv',
        'Premier League': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv'
    }
    df = pd.read_csv(file_paths[league])
    df['HomeTeamGoals'] = df['FTHG']
    df['AwayTeamGoals'] = df['FTAG']
    return df

# Poisson Model for Home and Away Teams
def predict_match_outcome(home_team, away_team, avg_goals_scored, avg_goals_conceded):
    home_team_offensive_strength = avg_goals_scored[home_team] / np.mean(list(avg_goals_scored.values()))
    away_team_defensive_strength = avg_goals_conceded[away_team] / np.mean(list(avg_goals_conceded.values()))
    home_goals = poisson.pmf(np.arange(0, 10), home_team_offensive_strength)
    away_goals = poisson.pmf(np.arange(0, 10), away_team_defensive_strength)
    match_matrix = np.outer(home_goals, away_goals)
    home_win_prob = np.sum(np.tril(match_matrix, -1))
    draw_prob = np.sum(np.diag(match_matrix))
    away_win_prob = np.sum(np.triu(match_matrix, 1))
    return home_win_prob, draw_prob, away_win_prob

import itertools


def predict_final_league_table(df, avg_goals_scored, avg_goals_conceded):
    # Initialize league table
    teams = df['HomeTeam'].unique()
    league_table = pd.DataFrame(index=teams, columns=['Points', 'GoalsFor', 'GoalsAgainst', 'GoalDifference', 'Movement'])
    league_table.fillna(0, inplace=True)

    # Set of played matches
    played_matches = set(zip(df['HomeTeam'], df['AwayTeam']))

    # Process played matches in the dataset
    for _, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        # Update league table
        league_table.at[home_team, 'GoalsFor'] += home_goals
        league_table.at[away_team, 'GoalsAgainst'] += home_goals
        league_table.at[away_team, 'GoalsFor'] += away_goals
        league_table.at[home_team, 'GoalsAgainst'] += away_goals

        if home_goals > away_goals:
            league_table.at[home_team, 'Points'] += 3
        elif home_goals < away_goals:
            league_table.at[away_team, 'Points'] += 3
        else:
            league_table.at[home_team, 'Points'] += 1
            league_table.at[away_team, 'Points'] += 1

    # Calculate Goal Difference
    league_table['GoalDifference'] = league_table['GoalsFor'] - league_table['GoalsAgainst']

    # Current standings based on played matches
    current_standings = league_table.sort_values(by=['Points', 'GoalDifference', 'GoalsFor'], ascending=[False, False, False])

    # Predict outcomes for unplayed matches
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team and ((home_team, away_team) not in played_matches):
                home_win_prob, draw_prob, away_win_prob = predict_match_outcome(home_team, away_team, avg_goals_scored, avg_goals_conceded)
                # Decide the most likely outcome
                if home_win_prob > max(draw_prob, away_win_prob):
                    home_goals, away_goals = 1, 0  # Home win
                elif away_win_prob > max(draw_prob, home_win_prob):
                    home_goals, away_goals = 0, 1  # Away win
                else:
                    home_goals, away_goals = 0, 0  # Draw

                # Update the league table
                league_table.at[home_team, 'GoalsFor'] += home_goals
                league_table.at[away_team, 'GoalsAgainst'] += home_goals
                league_table.at[away_team, 'GoalsFor'] += away_goals
                league_table.at[home_team, 'GoalsAgainst'] += away_goals

                if home_goals > away_goals:
                    league_table.at[home_team, 'Points'] += 3
                elif home_goals < away_goals:
                    league_table.at[away_team, 'Points'] += 3
                else:
                    league_table.at[home_team, 'Points'] += 1
                    league_table.at[away_team, 'Points'] += 1

    # Sort final league table
    final_table = league_table.sort_values(by=['Points', 'GoalDifference', 'GoalsFor'], ascending=[False, False, False])

    # Calculate movement
    for team in league_table.index:
        current_pos = list(current_standings.index).index(team) + 1
        final_pos = list(final_table.index).index(team) + 1
        movement = 'Stable'
        if final_pos < current_pos:
            movement = 'Up'
        elif final_pos > current_pos:
            movement = 'Down'
        final_table.at[team, 'Movement'] = movement

    return final_table



# Use this function in your Streamlit app as needed



# Streamlit interface
st.title("Football Match Outcome Predictor :soccer:")

# League selection and data loading
league = st.selectbox("Select League", ['Bundesliga', 'La Liga', '2. Bundesliga', 'Premier League'])
df = load_data(league)

# Calculate average goals scored and conceded by each team
avg_goals_scored = df.groupby('HomeTeam')['HomeTeamGoals'].mean().to_dict()
avg_goals_conceded = df.groupby('AwayTeam')['AwayTeamGoals'].mean().to_dict()

st.subheader(':one: Predict individual games', divider='rainbow')

# Dropdown menus for team selection
teams = sorted(df['HomeTeam'].unique())
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

# Prediction and plotting
if st.button(":arrows_counterclockwise: Predict Match Outcome"):
    #home_win, draw, away_win, home_goals_dist, away_goals_dist = predict_match_outcome(home_team, away_team, avg_goals_scored, avg_goals_conceded)

    home_win, draw, away_win = predict_match_outcome(home_team, away_team, avg_goals_scored, avg_goals_conceded)
    st.write(f"Predicted probabilities for {home_team} vs {away_team}:")
    st.write(f"Home Win: {home_win:.2f}, Draw: {draw:.2f}, Away Win: {away_win:.2f}")


st.subheader(':two: Predict final league table', divider='rainbow')

# Predicting the final league table
if st.button(":arrows_counterclockwise: Predict Final League Table"):
    final_table = predict_final_league_table(df, avg_goals_scored, avg_goals_conceded)
    st.write("Predicted Final League Table:")
    st.dataframe(final_table)
