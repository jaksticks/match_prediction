import pandas as pd
import numpy as np
import datetime as dt
import re
from pathlib import Path
import logging
import penaltyblog as pb
from types import SimpleNamespace
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def initialize_webdriver(url):

    # initialize webdriver
    driver = webdriver.Chrome()
    # load the website
    driver.get(url)

    return driver

def scroll_page(driver):

    # Scroll down repeatedly (to load more content)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for loading

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # No more content to load
        last_height = new_height


def get_results(driver):

    # Get teams
    #elements = driver.find_elements(By.CLASS_NAME, 'teamname')
    elements = driver.find_elements(By.CSS_SELECTOR, ".matchrow.status-Played .teamname")
    teams = [elem.text for elem in elements]

    # Get goals
    #elements = driver.find_elements(By.CLASS_NAME, 'scorefs')
    elements = driver.find_elements(By.CSS_SELECTOR, ".matchrow.status-Played .scorefs")
    goals = [elem.text for elem in elements]

    # Check if the number of teams and goals match and only use games where the result is known
    #teams = teams[-len(goals):]
    assert len(teams) == len(goals), "Number of teams and goals do not match"

    # create dataframe
    df = pd.DataFrame(columns=['team_home', 'team_away', 'goals_home', 'goals_away'])
    df.team_home = teams[0::2]
    df.team_away = teams[1::2]
    df.goals_home = goals[0::2]
    df.goals_away = goals[1::2]

    return df

def get_fixtures(driver):
    
    # get teams
    elements = driver.find_elements(By.CLASS_NAME, 'teamname')
    teams = [elem.text for elem in elements]

    # re-create fixtures
    fixtures = pd.DataFrame(columns=['Home', 'Away'])
    fixtures.Home = teams[0::2]
    fixtures.Away = teams[1::2]
    
    return fixtures

def get_table(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')

    parsed_rows = []
    for row in rows:
        cells = [cell.text.strip() for cell in row.find_elements(By.CSS_SELECTOR, 'th, td')]
        if not cells:
            continue

        rank_index = next((index for index, value in enumerate(cells) if value.isdigit()), None)
        if rank_index is None:
            continue

        squad_index = rank_index + 1
        stats_index = squad_index + 1
        if len(cells) <= stats_index + 5:
            continue

        rank = cells[rank_index]
        squad = cells[squad_index]
        gf_ga_parts = re.split(r'\s*[\-–]\s*', cells[stats_index + 4])
        if len(gf_ga_parts) != 2:
            continue

        parsed_rows.append(
            {
                'Rk': rank,
                'Squad': squad,
                'MP': cells[stats_index],
                'W': cells[stats_index + 1],
                'D': cells[stats_index + 2],
                'L': cells[stats_index + 3],
                'GF': gf_ga_parts[0],
                'GA': gf_ga_parts[1],
                'GD': int(gf_ga_parts[0]) - int(gf_ga_parts[1]),
                'Pts': cells[stats_index + 5],
            }
        )

    table = pd.DataFrame(parsed_rows, columns=['Rk', 'Squad', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts'])

    if not table.empty:
        table[['Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']] = (
            table[['Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']].apply(pd.to_numeric, errors='coerce')
        )

    return table