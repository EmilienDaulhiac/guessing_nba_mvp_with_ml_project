from datetime import datetime
from dataclasses import dataclass
from urllib import parse
from bs4 import BeautifulSoup
from .html_scraper import HTMLScraper
import re
import pandas as pd

@dataclass
class BaskRefPlayerDataScraper(HTMLScraper):
    """Class for scraping & Parsing basketball-reference.com data"""

    # public functions

    def scrape_table_data(self, game_url: str, div_to_look_for: str, class_to_look_for: str = None) -> pd.DataFrame:
        """
        Scrapes the stats data for the given web page.
        :return: returns a pd.DataFrame of players stats
        """
        
        game_data = self.scrape(game_url, self._parse_table, div_to_look_for, class_to_look_for)

        return game_data

    def _parse_table(
        self, html: BeautifulSoup, div_to_look_for: str, class_to_look_for: str = None
    ) -> pd.DataFrame:
        """
        Provided the data page, it parses out
        the table and return a dataframe.
        :return: pd.Daframe
        """

        table_div = html.select_one('div#' + div_to_look_for)

        if table_div:
            # Extract relevant information from the selected div

            # Find the position element
            table_element = table_div.select_one('tbody')
            
            data = []
            if class_to_look_for:
                for pos, el in enumerate(table_element.find_all('tr', class_=class_to_look_for)):
                    data.append({})
                    for el2 in el.findAll('td'):
                        if el2.get('data-stat') != 'DUMMY':
                            try:
                                data[pos][el2.get('data-stat')] = float(el2.text)
                            except ValueError:
                                data[pos][el2.get('data-stat')] = el2.text
                            
            else:
                for pos, el in enumerate(table_element.find_all('tr')):
                    data.append({})
                    for el2 in el.findAll('td'):
                        if el2.get('data-stat') != 'DUMMY':
                            try:
                                data[pos][el2.get('data-stat')] = float(el2.text)
                            except ValueError:
                                data[pos][el2.get('data-stat')] = el2.text
            
            return pd.DataFrame(data)

        else:
            print("No div with id='info' found.")
            return None
    