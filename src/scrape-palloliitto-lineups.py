import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def initialize_webdriver(url: str, headless: bool = False) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    return driver


def scroll_page(driver: webdriver.Chrome, sleep_seconds: float = 2.0) -> None:
    import time

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_seconds)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def get_match_ids_from_results(results_url: str, headless: bool, scroll_sleep: float) -> list[str]:
    driver = initialize_webdriver(results_url, headless=headless)
    try:
        scroll_page(driver, sleep_seconds=scroll_sleep)
        elements = driver.find_elements(By.XPATH, "//*[@matchid]")
        match_ids = [el.get_attribute("matchid") for el in elements if el.get_attribute("matchid")]

        # Keep insertion order while deduplicating.
        unique_match_ids = list(dict.fromkeys(match_ids))
        return unique_match_ids
    finally:
        driver.quit()


def parse_score(score_text: str) -> tuple[Optional[int], Optional[int]]:
    first_line = score_text.strip().split("\n")[0]
    parts = re.split(r"\s*[\-–]\s*", first_line)
    if len(parts) < 2:
        return None, None

    try:
        home_goals = int(parts[0])
        away_goals = int(parts[1])
    except ValueError:
        return None, None

    return home_goals, away_goals


def extract_player_info_from_row(row, shirt_number: str) -> tuple[str, bool, bool]:
    selectors = [
        './/span[contains(@class, "namenarrow")]',
        './/span[contains(@class, "playername")]',
        './/span[contains(@class, "undefined")]',
    ]

    for selector in selectors:
        elems = row.find_elements(By.XPATH, selector)
        if elems:
            text = elems[0].text.strip()
            if text:
                break
    else:
        text = row.text.strip()

    # Parse role suffixes often shown as "Name | C" / "Name | MV" / "Name | MV | C".
    parts = [p.strip() for p in text.split("|")]
    name = parts[0] if parts else ""
    role_text_upper = " ".join(parts[1:]).upper()

    # Keep role checks independent so the same player can be both GK and captain.
    is_goalkeeper = bool(
        re.search(r"(^|[^A-Z0-9])(MV|GK|MAALIVAHTI|GOALKEEPER)([^A-Z0-9]|$)", role_text_upper)
    )
    is_captain = bool(
        re.search(r"(^|[^A-Z0-9])(C|CAPTAIN|KAPTEENI)([^A-Z0-9]|$)", role_text_upper)
    )

    if shirt_number and name.startswith(shirt_number):
        name = name[len(shirt_number) :].strip()

    name = re.sub(r"\s+\d+[×x]\s*$", "", name).strip()

    return name, is_goalkeeper, is_captain


def scrape_match_lineups(driver: webdriver.Chrome, match_id: str, timeout: int) -> list[dict]:
    url = f"https://tulospalvelu.palloliitto.fi/match/{match_id}/lineups"
    driver.get(url)

    wait = WebDriverWait(driver, timeout)
    wait.until(EC.presence_of_element_located((By.ID, "matchscore")))

    home_team = driver.find_element(By.CSS_SELECTOR, "#A_team .teamname").text.strip()
    away_team = driver.find_element(By.CSS_SELECTOR, "#B_team .teamname").text.strip()

    score_text = driver.find_element(By.ID, "matchscore").text
    home_goals, away_goals = parse_score(score_text)

    player_tables = driver.find_elements(By.XPATH, '//div[contains(@class, "playerlist")]')
    teams = [home_team, away_team]
    goals_for = [home_goals, away_goals]
    goals_against = [away_goals, home_goals]

    players: list[dict] = []

    for idx, table in enumerate(player_tables[:2]):
        team_label = teams[idx] if idx < len(teams) else ""

        rows = table.find_elements(By.XPATH, ".//table//tr")
        for row in rows:
            shirt_elems = row.find_elements(By.XPATH, './/span[contains(@class, "shirtnumber")]')
            shirt_number = shirt_elems[0].text.strip() if shirt_elems else ""

            player_name, is_goalkeeper, is_captain = extract_player_info_from_row(
                row, shirt_number=shirt_number
            )
            # Skip non-player rows.
            if not player_name or player_name.lower() in {"vaihtopelaajat", "toimihenkilot", "toimihenkilot"}:
                continue

            players.append(
                {
                    "match_id": match_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team_label,
                    "shirt_number": shirt_number,
                    "player_name": player_name,
                    "is_goalkeeper": is_goalkeeper,
                    "is_captain": is_captain,
                    "team_goals_scored": goals_for[idx] if idx < len(goals_for) else None,
                    "team_goals_conceded": goals_against[idx] if idx < len(goals_against) else None,
                    "lineups_url": url,
                }
            )

    return players


def scrape_lineups(results_url: str, headless: bool, timeout: int, scroll_sleep: float, limit: Optional[int]) -> pd.DataFrame:
    match_ids = get_match_ids_from_results(results_url, headless=headless, scroll_sleep=scroll_sleep)
    if limit is not None:
        match_ids = match_ids[:limit]

    logging.info("Found %s match ids", len(match_ids))

    all_rows: list[dict] = []
    driver = initialize_webdriver("about:blank", headless=headless)
    try:
        for i, match_id in enumerate(match_ids, start=1):
            logging.info("Scraping lineups for match %s/%s (id=%s)", i, len(match_ids), match_id)
            try:
                rows = scrape_match_lineups(driver, match_id=match_id, timeout=timeout)
            except TimeoutException:
                logging.warning("Skipping match %s due to timeout", match_id)
                continue
            except Exception as exc:
                logging.warning("Skipping match %s due to error: %s", match_id, exc)
                continue

            all_rows.extend(rows)
    finally:
        driver.quit()

    return pd.DataFrame(all_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape team lineups from Palloliitto Tulospalvelu")
    parser.add_argument(
        "--results-url",
        required=True,
        help="Results page URL, for example: https://tulospalvelu.palloliitto.fi/category/M7!etejp25/results",
    )
    parser.add_argument(
        "--output",
        default="../data/palloliitto_lineups.csv",
        help="Output CSV path",
    )
    parser.add_argument("--timeout", type=int, default=10, help="Element wait timeout in seconds")
    parser.add_argument("--scroll-sleep", type=float, default=2.0, help="Seconds to sleep between scroll steps")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of matches to scrape")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()

    df = scrape_lineups(
        results_url=args.results_url,
        headless=args.headless,
        timeout=args.timeout,
        scroll_sleep=args.scroll_sleep,
        limit=args.limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logging.info("Saved %s lineup rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
