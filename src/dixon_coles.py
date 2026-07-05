import argparse
import logging

import penaltyblog as pb
import pandas as pd
import numpy as np
import datetime as dt
import dataframe_image as dfi
from random import choices
import json
import requests
from tqdm import tqdm
import time
import pytz
import soccerdata as sd
from utils import (simulate_match, analyse_match, table_bonus_check, calculate_manager_points, simulate_season)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def generate_upcoming_fixtures(matches, all_teams):
    """
    Generate upcoming fixtures by calculating all possible matchups for the season
    and removing games that have already been played.
    
    Parameters:
    -----------
    matches : pd.DataFrame
        DataFrame containing all matches (played and unplayed)
    
    Returns:
    --------
    upcoming_fixtures : pd.DataFrame
        DataFrame containing only the upcoming fixtures
    """
    
    logging.info("Generating upcoming fixtures by deducing season schedule...")
    
    # Generate all possible matchups (each team plays each other team twice)
    all_possible_fixtures = []
    for home_team in all_teams:
        for away_team in all_teams:
            if home_team != away_team:
                all_possible_fixtures.append({
                    'home_team': home_team,
                    'away_team': away_team
                })
    
    all_fixtures_df = pd.DataFrame(all_possible_fixtures)
    logging.info(f"Generated {len(all_fixtures_df)} possible matchups")
    
    # Get played matches
    matches_aux = matches.reset_index()
    current_season = matches_aux['season'].max()
    # keep games from current season
    played_matches = matches_aux[matches_aux['season']==current_season].copy()
    # keep only games that have been played
    played_matches = played_matches[played_matches['home_score'].notnull()]
    # keep only home_team and away_team columns
    played_matchups = played_matches[['home_team', 'away_team']]
    logging.info(f"Found {len(played_matchups)} played matchups")

    # Find upcoming fixtures by removing played matches from all possible fixtures
    upcoming_fixtures = all_fixtures_df.merge(
        played_matchups,
        on=['home_team', 'away_team'],
        how='left',
        indicator=True
    )
    upcoming_fixtures = upcoming_fixtures[upcoming_fixtures['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    logging.info(f"Generated {len(upcoming_fixtures)} upcoming fixtures")
    
    # Keep only the columns that exist in the original matches DataFrame
    # and fill with NaN for missing columns
    for col in matches.columns:
        if col not in upcoming_fixtures.columns:
            upcoming_fixtures[col] = None
    
    upcoming_fixtures = upcoming_fixtures[matches.columns]
    
    return upcoming_fixtures

def fetch_data(league_name, seasons:list, deduce_schedule=False):
    """
    Fetch data for the specified league.
    """

    logging.info(f"Fetching data for seasons {seasons} of {league_name}...")
    sofascore = sd.Sofascore(leagues=league_name, seasons=seasons)
    matches = sofascore.read_schedule()

    logging.info(f"Fetching league table for the current season ({seasons[-1]}) of {league_name}...")
    sofascore = sd.Sofascore(leagues=league_name, seasons=seasons[-1])
    league_table = (sofascore.read_league_table()
            .rename(columns={'team': 'Squad'})
            .reset_index(drop=True)
        )
    league_table['Rk'] = league_table.index + 1
    league_table['GF'] = league_table['GF'].astype(int)
    league_table['GA'] = league_table['GA'].astype(int)
    teams = np.sort(league_table['Squad'].unique())
    
    # get fixtures for the rest of the season
    if deduce_schedule:
        fixtures = generate_upcoming_fixtures(matches, teams)
        logging.info(f"Generated upcoming fixtures using deduced schedule")
    else:
        fixtures = matches[matches.home_score.isnull()].copy()  # matches that have not been played yet
        logging.info(f"Using fixtures from API")

    logging.info(f"Processing data {league_name} data...")
    df = matches[matches['home_score'].notnull()].copy()
    df.rename(columns={'home_team': 'team_home', 'away_team': 'team_away', 'home_score': 'goals_home', 'away_score': 'goals_away'}, inplace=True)
    df = df.sort_values('date').reset_index(drop=True)
    current_date = dt.datetime.now(pytz.UTC)
    df['days_since'] = df['date'].apply(lambda x: (current_date-x).days)
    df = df[df.days_since <= 850].copy() # only keep matches within the last 850 days
    logging.info(f"Results retrieved up to {df['date'].max().date()}")
    
    return df, fixtures, league_table, teams

    
def create_model(df):
    """
    Fit the Dixon-Coles model.
    """

    logging.info("Creating the Dixon-Coles model...")
    xi = 0.0018
    weights = pb.models.dixon_coles_weights(df["date"], xi)

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )
    clf.fit()

    return clf

def get_median_from_list(numbers):
    # Sort the list
    sorted_numbers = sorted(numbers)
    # Get the middle index
    middle_index = len(sorted_numbers) // 2
    # Return the middle value
    return sorted_numbers[middle_index]

def create_team_ratings(clf, teams, args, export_ratings=True):
    """
    Create team ratings based on the fitted model.
    """
    logging.info("Creating team ratings...")
    
    # Extract attack and defense parameters
    params = clf.get_params()
    attack_params = {k: v for k, v in params.items() if k.startswith('attack_')}
    defense_params = {k: v for k, v in params.items() if k.startswith('defence_')}

    # Compute median values
    median_attack = get_median_from_list(list(attack_params.values()))
    median_defense = get_median_from_list(list(defense_params.values()))
    
    # Find teams with median values
    median_attack_team = [team.split('attack_')[1] for team, value in attack_params.items() if value == median_attack]
    median_defense_team = [team.split('defence_')[1] for team, value in defense_params.items() if value == median_defense]

    logging.info(f"Median attack value: {median_attack}, Team(s): {median_attack_team}")
    logging.info(f"Median defense value: {median_defense}, Team(s): {median_defense_team}")

    ratings = []
    for team in teams:
        team_attack_rating_home =  clf.predict(team, median_defense_team[0]).home_goal_expectation
        team_attack_rating_away = clf.predict(median_defense_team[0], team).away_goal_expectation
        team_attack_rating = np.mean((team_attack_rating_home, team_attack_rating_away))
        
        team_defense_rating_home = clf.predict(team, median_attack_team[0]).away_goal_expectation
        team_defense_rating_away = clf.predict(median_attack_team[0], team).home_goal_expectation
        team_defense_rating = np.mean((team_defense_rating_home, team_defense_rating_away))
        
        team_goal_difference_rating = team_attack_rating - team_defense_rating
        ratings.append((team, team_attack_rating, team_defense_rating, team_goal_difference_rating))

    ratings_df = (pd.DataFrame(
        ratings, 
        columns=['team', 'attack_rating', 'defense_rating', 'goal_difference_rating'])
        .sort_values(by='goal_difference_rating', ascending=False).reset_index(drop=True)
    )
    ratings_df.index += 1

    # Log the DataFrame
    logging.info("Team ratings:\n%s", ratings_df.to_string(index=True))

    if export_ratings:
        league_name = args.league.replace(" ", "_")
        dfi.export(ratings_df, f"../output/ratings_{league_name}.png", table_conversion='matplotlib',)

    return ratings_df
    

def simulate_league(clf, league_table, fixtures, nr_simulations):
    """
    Simulate the season using the fitted model.
    """

    logging.info("Starting to run simulations...")
    simulation_results = []
    for i in tqdm(range(nr_simulations), desc='Simulating...'):
        simulated_table = league_table[['Rk', 'Squad', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']].copy()
        simulated_table = simulate_season(simulated_table, fixtures, clf)
        simulated_table['simulation_nr'] = i
        simulation_results.append(simulated_table)

    logging.info("Simulations finished.")
    simulation_results_df = pd.concat(simulation_results).reset_index(drop=True)
    return simulation_results_df


def process_simulation_results(simulation_results_df, nr_simulations, args):
    """
    Process and analyze the simulation results.
    """

    logging.info("Processing simulation results...")
    
    max_rank = simulation_results_df['Rk'].max()
    result_matrix = (
        simulation_results_df.groupby(['Squad', 'Rk'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, max_rank+1), fill_value=0)  # Ensure columns go from 1 to max_rank
    )

    # normalize to percentage
    result_matrix = 100 * (result_matrix / nr_simulations)

    # Reorder the matrix based on average final league position
    sorted_teams = simulation_results_df.groupby(['Squad'])['Rk'].mean().sort_values().index
    sorted_matrix = result_matrix.loc[sorted_teams]
    
    if args.save_simulation_results:
        # Get the current timestamp in YY-MM-DD_HR-MIN-SEC format
        timestamp = dt.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        # Use the timestamp as part of the file name
        league_name = args.league.replace(" ", "_")
        sorted_matrix.to_csv(f'../output/league_matrices/{timestamp}--{league_name}.csv')

    # Plot the reordered heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(sorted_matrix, annot=True, cmap="Blues", linewidths=0.5, cbar_kws={'label': 'Probability'})

    timestamp = dt.datetime.now().strftime("%d.%m.%Y")
    plt.title(f"Projected Final League Positions {args.league} \n Forecast on {timestamp}")
    plt.xlabel("Final League Position")
    plt.ylabel("Team")

    if args.save_simulation_results:
        league_name = args.league.replace(" ", "_")
        plt.savefig(f'../output/league_distribution_{league_name}.png')

    #plt.show()


def main(args, seasons):
    """
    Fit Dixon-Coles model to the specified league and simulate match outcomes.
    """
    logging.info(f"Creating a Dixon-Coles model for {args.league}")
    
    df, fixtures, league_table, teams = fetch_data(args.league, seasons, args.deduce_schedule)
    clf = create_model(df)
    create_team_ratings(clf, teams, args)
    simulation_results_df = simulate_league(clf, league_table, fixtures, args.nr_simulations)
    process_simulation_results(simulation_results_df, args.nr_simulations, args)
    
    logging.info("Done.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fit a Dixon-Coles model for a specific league and simulate final league table standings.")
    parser.add_argument(
        "--league",
        type=str,
        default="ENG-Premier League",
        choices=["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"],
        help="Name of the league to process (e.g., 'ENG Premier League'). Default is 'ENG Premier League'."
    )
    parser.add_argument(
        "--nr_simulations",
        type=int,
        default=1000,        
        help="Number of simulations to run. Default is 1000."
    )
    parser.add_argument(
        "--save_simulation_results",
        type=bool,
        default=True,
        help="Whether to save the simulation results. Default is True."        
    )
    parser.add_argument(
        "--deduce_schedule",
        type=bool,
        default=False,
        help="Whether to deduce the season schedule to generate upcoming fixtures. Default is False (use API fixtures)."
    )
    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Read the config file
    config_file_path = "config"
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    seasons = config["seasons"]

    # Call the main function 
    main(args, seasons)