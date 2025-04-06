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
from utils import (simulate_match, analyse_match, table_bonus_check, calculate_manager_points, simulate_season)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def fetch_data(league_name, urls):
    """
    Fetch data for the specified league.
    """

    logging.info(f"Fetching data for {league_name}...")
    matches_season = pd.read_html(urls['url_current_season'])[0]
    matches_previous_season = pd.read_html(urls['url_previous_season'])[0]
    fixtures = matches_season[(matches_season.Home.notnull()) & (matches_season.Score.isnull())]
    league_table = pd.read_html(urls['url_league_table'])[0]
    teams = np.sort(league_table['Squad'].unique())
    
    logging.info(f"Processing data {league_name} data...")
    matches = pd.concat([matches_season, matches_previous_season], ignore_index=True)
    matches['Date'] = pd.to_datetime(matches['Date'])
    df = matches[matches['Score'].notnull()]
    df = df.reset_index()
    df['goals_home'] = df['Score'].apply(lambda x: x.split('–')[0])
    df['goals_away'] = df['Score'].apply(lambda x: x.split('–')[1])
    df.rename(columns={'Home': 'team_home', 'Away': 'team_away', 'Date': 'date'}, inplace=True)

    return df, fixtures, league_table, teams

    
def create_model(df):
    """
    Fit the Dixon-Coles model.
    """

    logging.info("Creating the Dixon-Coles model...")
    xi = 0.001
    weights = pb.models.dixon_coles_weights(df["date"], xi)

    clf = pb.models.DixonColesGoalModel(
        df["goals_home"], df["goals_away"], df["team_home"], df["team_away"], weights
    )
    clf.fit()

    return clf


def create_team_ratings(clf, teams, args):
    """
    Create team ratings based on the fitted model.
    """
    logging.info("Creating team ratings...")
    
    # Extract attack and defense parameters
    params = clf.get_params()
    attack_params = {k: v for k, v in params.items() if k.startswith('attack_')}
    defense_params = {k: v for k, v in params.items() if k.startswith('defence_')}

    # Compute median values
    median_attack = np.median(list(attack_params.values()))
    median_defense = np.median(list(defense_params.values()))

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

    league_name = args.league.replace(" ", "_")
    dfi.export(ratings_df, f"../output/ratings_{league_name}.png", table_conversion='matplotlib',)
    

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
    
    result_matrix = (
        simulation_results_df.groupby(['Squad', 'Rk'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 21), fill_value=0)  # Ensure columns go from 1 to 20
    )

    # normalize to percentage
    result_matrix = 100 * (result_matrix / nr_simulations)

    # Reorder the matrix based on average final league position
    sorted_teams = simulation_results_df.groupby(['Squad'])['Rk'].mean().sort_values().index
    sorted_matrix = result_matrix.loc[sorted_teams]

    # Plot the reordered heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(sorted_matrix, annot=True, cmap="Blues", linewidths=0.5, cbar_kws={'label': 'Probability'})

    timestamp = dt.datetime.now().strftime("%d.%m.%Y")
    plt.title(f"Projected Final League Positions {args.league} \n Forecast on {timestamp}")
    plt.xlabel("Final League Position")
    plt.ylabel("Team")

    if args.save_simulation_results:
        league_name = args.league.replace(" ", "_")
        plt.savefig(f'../output/league_distribution_{league_name}.png')

    plt.show()


def main(args, urls):
    """
    Fit Dixon-Coles model to the specified league and simulate match outcomes.
    """
    logging.info(f"Creating a Dixon-Coles model for {args.league}")
    
    df, fixtures, league_table, teams = fetch_data(args.league, urls)
    clf = create_model(df)
    create_team_ratings(clf, teams, args)
    simulation_results_df = simulate_league(clf, league_table, fixtures, args.nr_simulations)
    process_simulation_results(simulation_results_df, args.nr_simulations, args)
    
    logging.info("Done.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Dixon-Coles model for a specific league.")
    parser.add_argument(
        "--league",
        type=str,
        default="ENG Premier League",
        choices=["ENG Premier League", "ESP La Liga"],
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
        help="Whether to save the simulation results as an image. Default is True."        
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
    # Get urls from the config file
    urls = config[args.league]

    # Call the main function 
    main(args, urls)