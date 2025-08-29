import pandas as pd
import numpy as np
import datetime as dt
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

    # get teams
    elements = driver.find_elements(By.CLASS_NAME, 'text-start')
    teams = [elem.text for elem in elements]
    teams = teams[7::5]

    # get stats
    elements = driver.find_elements(By.CLASS_NAME, 'text-center')
    stats = [elem.text for elem in elements]
    stats = stats[6::]

    table = pd.DataFrame(columns=['Rk', 'Squad', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts'])
    table.Rk = np.arange(1, len(teams)+1)
    table.Squad = teams
    table.MP = stats[0::6]
    table.W = stats[1::6]
    table.D = stats[2::6]
    table.L = stats[3::6]
    table.GF = [x.split('–')[0] for x in stats[4::6]]
    table.GA = [x.split('–')[1] for x in stats[4::6]]
    table.GD = table['GF'].astype(int) - table['GA'].astype(int)
    table.Pts = stats[5::6]

    # convert to numeric
    table[['Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']] = (
        table[['Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']].apply(pd.to_numeric, errors='coerce')
    )

    return table