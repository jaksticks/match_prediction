import penaltyblog as pb
import pandas as pd
import numpy as np
import datetime as dt
import dataframe_image as dfi
from random import choices
import seaborn as sns

def simulate_match(clf, homeTeam, awayTeam):
    '''
    Simulate the outcome of a single match.
    '''

    probs = clf.predict(homeTeam, awayTeam)

    # simulate scoreline
    marginal_home = np.sum(probs.grid, axis=1)
    marginal_away = np.sum(probs.grid, axis=0)
    home_goals = choices(np.arange(15), weights=marginal_home)[0]
    away_goals = choices(np.arange(15), weights=marginal_away)[0]

    if home_goals > away_goals:
        outcome = 'home_win'
    elif home_goals == away_goals:   
        outcome = 'draw'
    else:
        outcome = 'away_win'

    return outcome, home_goals, away_goals

def analyse_match(clf, homeTeam, awayTeam):

    probs = clf.predict(homeTeam, awayTeam)
    display(probs)

    ax = sns.heatmap(probs.grid[:6, :6], annot=True, fmt=".2f")
    ax.set(xlabel=awayTeam, ylabel=homeTeam)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')


def table_bonus_check(focal_team, opponent, league_table):
    focal_team_rank = league_table.loc[league_table.Squad==focal_team, 'Rk'].values[0]
    opponent_rank = league_table.loc[league_table.Squad==opponent, 'Rk'].values[0]
    if focal_team_rank - opponent_rank >= 5:
        return 1.0
    else:
        return 0.0
    
def calculate_manager_points(outcome, home_goals, away_goals, home_team, away_team, gameweek, league_table, manager_xp):
    # CALCULATE EXPECTED MANAGER POINTS FOR HOME TEAM
    new_row1 = [gameweek, home_team, 0]
    table_bonus_coefficient = table_bonus_check(home_team, away_team, league_table) # 1 or 0
    # points for win
    if outcome=='home_win':
        new_row1[2] = (6 + 10*table_bonus_coefficient)
    # points for draw
    elif outcome=='draw':
        new_row1[2] = (3 + 5*table_bonus_coefficient)
    # goals scored
    new_row1[2] += home_goals
    # points for clean sheet
    if away_goals==0:
        new_row1[2] += 2

    # CALCULATE EXPECTED MANAGER POINTS FOR AWAY TEAM
    new_row2 = [gameweek, away_team, 0]
    table_bonus_coefficient = table_bonus_check(away_team, home_team, league_table) # 1 or 0
    # points for win
    if outcome=='away_win':
        new_row2[2] += (6 + 10*table_bonus_coefficient)
    # points for draw
    elif outcome=='draw':
        new_row2[2] += (3 + 5*table_bonus_coefficient)
    # goals scored
    new_row2[2] += away_goals
    # points for clean sheet
    if home_goals==0:
        new_row2[2] += 2

    # add rows to manager_xp
    manager_xp.extend([new_row1, new_row2])

    return manager_xp

def update_league_table(league_table, home_team_, away_team_, outcome, home_goals, away_goals, fixture):
    '''Update league table after each match.'''
    # update matches played
    league_table.loc[league_table['Squad']==home_team_, 'MP'] += 1
    league_table.loc[league_table['Squad']==away_team_, 'MP'] += 1
    # update league table 
    league_table.loc[league_table['Squad']==home_team_, 'GF'] += home_goals
    league_table.loc[league_table['Squad']==home_team_, 'GA'] += away_goals
    league_table.loc[league_table['Squad']==home_team_, 'GD'] += home_goals - away_goals
    league_table.loc[league_table['Squad']==away_team_, 'GF'] += away_goals
    league_table.loc[league_table['Squad']==away_team_, 'GA'] += home_goals
    league_table.loc[league_table['Squad']==away_team_, 'GD'] += away_goals - home_goals
    if outcome=='home_win':
        league_table.loc[league_table['Squad']==home_team_, 'Pts'] += 3
        league_table.loc[league_table['Squad']==home_team_, 'W'] += 1
        league_table.loc[league_table['Squad']==away_team_, 'L'] += 1
    elif outcome=='draw':
        league_table.loc[league_table['Squad']==home_team_, 'Pts'] += 1
        league_table.loc[league_table['Squad']==away_team_, 'Pts'] += 1
        league_table.loc[league_table['Squad']==home_team_, 'D'] += 1
        league_table.loc[league_table['Squad']==away_team_, 'D'] += 1
    elif outcome=='away_win':
        league_table.loc[league_table['Squad']==away_team_, 'Pts'] += 3
        league_table.loc[league_table['Squad']==away_team_, 'W'] += 1
        league_table.loc[league_table['Squad']==home_team_, 'L'] += 1
    else: 
        print('No valid result for:')
        print(fixture)

def simulate_season(league_table, fixtures, clf):
    '''Simulate a whole season.'''
    
    for _, fixture in fixtures.iterrows():
        home_team_ = fixture.Home
        away_team_ = fixture.Away
        # simulate match outcome
        outcome, home_goals, away_goals = simulate_match(clf, home_team_, away_team_)
        update_league_table(league_table, home_team_, away_team_, outcome, home_goals, away_goals, fixture)
    
    assert np.all(league_table['MP']==38), 'All teams have not played 38 games!'

    league_table = league_table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    league_table['Rk'] = np.arange(1,21)

    return league_table


def simulate_season_fpl(league_table, fixtures, clf, final_gameweek=38):
    '''Simulate a whole FPL season or up to certain gameweek.'''
    first_index = fixtures.event.first_valid_index()
    current_week = fixtures.loc[first_index,'event'].copy()
    league_table_snapshot = league_table.copy()
    fixtures = fixtures[fixtures.event<=final_gameweek].copy()
    manager_xp = []
    for _, fixture in fixtures.iterrows():
        fixture_week = fixture.event
        if np.isnan(fixture_week):
            continue
        elif fixture_week == current_week:
            pass  
        elif fixture_week == (current_week + 1):
            league_table = league_table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
            league_table['Rk'] = np.arange(1,21)
            league_table_snapshot = league_table.copy()
            current_week += 1
        else:
            print('Incorrect gameweek!')
            break

        home_team_ = fixture.home_team
        away_team_ = fixture.away_team
        # simulate match outcome
        outcome, home_goals, away_goals = simulate_match(clf, home_team_, away_team_)

        # calculate manager points
        manager_xp = calculate_manager_points(outcome, home_goals, away_goals, home_team_, away_team_, current_week, league_table_snapshot, manager_xp)

        # update league table
        update_league_table(league_table, home_team_, away_team_, outcome, home_goals, away_goals, fixture)

    if final_gameweek==38:
        assert np.all(league_table['MP']==38), 'All teams have not played 38 games!'

    league_table = league_table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    league_table['Rk'] = np.arange(1,21)

    return league_table, manager_xp

