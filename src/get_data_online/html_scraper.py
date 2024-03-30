from dataclasses import dataclass
from typing import Callable, Any
import requests
from requests import Response
from bs4 import BeautifulSoup

@dataclass
class HTMLScraper:
    """Class for scraping the web"""

    proxy: str | None = None

    def scrape(self, url: str, parser_fun: Callable, div_to_look_for: str, class_to_look_for: str) -> Any:
        """
        This function lays out the skeleton for scraping.
        First sends a GET request to the provided url and then uses
        the provided function to parse out the wanted data.
        """

        page = self.get_page(url)

        soup = BeautifulSoup(page.text, "html.parser")
        return parser_fun(soup, div_to_look_for, class_to_look_for)


    def get_page(self, url: str) -> Response:
        """
        This function uses as GET request 
        to scrape a static webpage from the web.
        """

        with requests.Session() as session:
            page = session.get(url, proxies=self._proxies())

        return page


    def _proxies(self) -> dict | None:
        """If the object is initialized with a proxy then use it"""
        if self.proxy:
            return {
                "https": self.proxy,
                "http": self.proxy,
            }
        return None