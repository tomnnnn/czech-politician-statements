import asyncio
import os
import sys
import json
import math
from tqdm.asyncio import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from .statement_parser import StatementParser
from .config import CONFIG
from .utils import fetch

class DemagogScraper:
    fetch_sem = asyncio.Semaphore(CONFIG['FetchConcurrency'])
    sleep_time = CONFIG['FetchDelay']
    base_url = "https://demagog.cz"
    stmts_url = "https://demagog.cz/vyroky"

    def __init__(self, from_year=None, to_year=None, first_npages=None):
        self.base_url = u'https://demagog.cz/vyroky?q=&years={}&page={}'
        self.fetch_sem = asyncio.Semaphore(CONFIG['FetchConcurrency'])
        self.first_npages = first_npages

        self.years = self.__loadYears()
        # filter the years
        from_year = from_year or min(self.years.keys())
        to_year = to_year or max(self.years.keys())

        self.years = {year: count for year, count in self.years.items() if from_year <= year <= to_year}

    def __loadYears(self) -> dict[int, int]:
        """Load the years from a file or scrape if not found."""
        path = f"{CONFIG['YearListPath']}"
        
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
                return {int(year): count for year, count in data.items()}


        # Scrape the years if the file is not found
        years = self.__scrapeYears()
        year_list_dir = os.path.dirname(CONFIG['YearListPath'])
        os.makedirs(year_list_dir, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(years, f, indent=2, ensure_ascii=False)
    
        return years

    def __scrapeYears(self):
        """
        Scrapes available years from the Demagog page

        Returns:
            list of (year, count) entries for each available year.
        """
        driver = webdriver.Chrome()
        driver.get(self.stmts_url)

        filters_button = driver.find_elements(By.CSS_SELECTOR, ".btn.w-100.h-44px")[0]
        filters_button.click()

        # get years filter
        filters = driver.find_elements(By.CSS_SELECTOR, "div.filter")
        year_filter = filters[-1]

        # scroll to the year filter
        driver.execute_script("arguments[0].scrollIntoView();", year_filter)

        actions = webdriver.ActionChains(driver)
        actions.move_to_element(year_filter)
        actions.perform()

        # expand years filter
        year_filter.click()

        # get the years
        year_divs = year_filter.find_elements(By.CSS_SELECTOR, ".filter-content")[0].find_elements(By.CSS_SELECTOR, ".check-btn")

        # parse the years
        years = {}
        for year_div in year_divs:
            year = int(year_div.find_elements(By.CSS_SELECTOR, ".small")[0].text.strip())
            count = int(
                year_div.find_elements(By.CSS_SELECTOR, ".smallest")[0]
                .text.split(' ')[0]
                .replace(',', '')
            )

            years[year] = count

        driver.quit()
        return years


    async def __get_statements(self, year, page, include_evidence) -> list:
        async with self.fetch_sem:
            url = self.base_url.format(year, page)
            html = await fetch(url)
            await asyncio.sleep(self.sleep_time)

        if not html:
            print(f"Failed to fetch {url}", file=sys.stderr)
            return []

        soup = BeautifulSoup(html, "html.parser")
        statement_divs = soup.find_all("div", class_="s-statement")

        statements = []
        corrupted_count = 0
        for statement_div in statement_divs:
            try:
                parsed = StatementParser(statement_div, include_evidence).get_dict()
                statements.append(parsed)
            except Exception as e:
                print(f"Failed to parse statement: {e}", file=sys.stderr)
                corrupted_count += 1

        if corrupted_count:
            print(f"Failed to parse {corrupted_count} statements", file=sys.stderr)

        return statements

    async def run(self, include_evidence=False) -> list:
        """
        Scrape statements for all years defined in the self.years dictionary.
        Each year corresponds to a number of pages to scrape.

        Args:
            include_evidence: whether to scrape evidence links for each statement
        """
        print("evidence", include_evidence)
        if self.first_npages:
            self.years = {year: min(count, self.first_npages*10) for year, count in self.years.items()}

        tasks = [
            self.__get_statements(year, page, include_evidence) 
            for year, count in self.years.items()
            for page in range(1, math.ceil(count/10)+1)
        ] 
        pages = await tqdm.gather(*tasks, desc="Scraping statements", unit="page", total=len(tasks))
        statements = [stmt for page in pages for stmt in page]

        # add ids to the statements
        statements = [{"id":i,**stmt} for i, stmt in enumerate(statements, 1)]

        return statements
