import asyncio
import os
import bs4
import json
import subprocess
from tqdm.asyncio import tqdm
from collections import defaultdict
import sys
from utils import fetch
from tqdm.asyncio import tqdm_asyncio
import uuid

class ArticleScraper:
    def __init__(self, fetch_concurrency=5, fetch_delay=0.5):
        self.fetch_sem = asyncio.Semaphore(fetch_concurrency)
        self.fetch_delay = fetch_delay

    def __extract_article(self, html, max_num_paragraphs=1000, min_num_words=30, output_html=True):
        """
        Extract article content from HTML using a combination of article tags and paragraph heuristics

        Args:
            html (str): The HTML content to extract the article from.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.
        """

        try:
            soup = bs4.BeautifulSoup(html, "html.parser", parse_only=bs4.SoupStrainer(["title","body"]))
        except Exception as e:
            raise e

        parsed_article = {
            "title": "",
            "content": ""
        }

        # if soup.find("article") is not None:
        #     article = soup.find("article")
        #     article_text = article.get_text()
        #     title = soup.find('title')
        #     title = title.get_text() if title else ""
        #
        #     parsed_article["title"] = title
        #     parsed_article["content"] = article_text
        #     return parsed_article

        # no article found, resort to paragraph heuristics
        p_parents = defaultdict(list)

        ps = soup.find_all("p")[:max_num_paragraphs]

        for p in ps:
            p_parents[p.parent].append(p)

        parents_counts = sorted([(parent, len(ps)) for parent, ps in p_parents.items()], key=lambda v: -v[1])
        if not parents_counts:
            raise ValueError("No paragraphs found")

        article_dom = parents_counts[0][0]

        if output_html:
            article_text = article_dom.decode_contents()
        else:
            article_text = "\n".join(p.get_text(strip=True) for p in article_dom.find_all("p") if len(p.get_text().split()) > min_num_words)

        title = soup.find('title')
        title = title.get_text() if title else ""

        parsed_article["title"] = title
        parsed_article["content"] = article_text

        return parsed_article


    async def batch_scrape(self, links: list[str], id=None, max_num_paragraphs: int=1000, min_num_words:int=5, show_progress=True, output_html=True):
        """
        Extract article content from a batch of URLs.

        Args:
            links (List[Dict[str, str]]): A list of dictionaries containing the URL and ID of the article.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.

        Returns:
            dict|list: If an ID is provided, a dictionary containing the ID and a list of articles. Otherwise, a list of articles.
        """
        tasks = [self.scrape(link, max_num_paragraphs, min_num_words, output_html) for link in links]
        articles = await tqdm.gather(*tasks, desc="Retrieving and extracting articles", unit="article", disable=not show_progress)
        articles = [r for r in articles if r is not None]

        return articles

    async def scrape(self, url, max_num_paragraphs=1000, min_num_words=5, output_html=True):
        """
        Fetch and extract article content from a URL

        Args:
            url (str): The URL to extract content from.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.
            id (str): The ID of the article.
        """
        async with self.fetch_sem:
            html = await fetch(url)
            if not html:
                print('Failed to fetch', url, file=sys.stderr)
                return None
            await asyncio.sleep(self.fetch_delay)

        try:
            article = self.__extract_article(html, max_num_paragraphs, min_num_words, output_html)
        except Exception as e:
            print(f"Failed to extract article from {url}: {e}", file=sys.stderr)
            return None

        result = {
            "url": url,
            **article,
        }

        return result

    @staticmethod
    async def scrape_extractus_async(links: list[str], show_progress=True):
        """
        Scrapes the articles using nodejs script with article-extractor library (asynchronously)
        """

        parser_path = os.path.join(os.path.dirname(__file__), "extractus/parse_article.js")

        fetch_tasks = [fetch(link) for link in links ]
        htmls = await tqdm_asyncio.gather(*fetch_tasks, desc="Fetching articles", unit="article", disable=not show_progress)


        async def run_parse_task(parser_path, html):
            cmd = ["node", parser_path]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
            )
            stdout, stderr = await process.communicate(input=html.encode("utf-8"))
            if process.returncode != 0:
                raise RuntimeError(f"Node script failed with error: {stderr.decode('utf-8')}")

            return json.loads(stdout.decode("utf-8"))

        parse_tasks = []
        for html in htmls:
            if html:
                parse_tasks.append(
                    run_parse_task(parser_path, html)
                )

        articles = await tqdm_asyncio.gather(*parse_tasks, desc="Parsing articles", unit="article", disable=not show_progress)
        articles = [article for article in articles if article is not None] # filter out None values

        return articles

    def scrape_extractus(self, links, id=None, max_num_paragraphs=1000, min_num_words=5, show_progress=True, output_html=True):
        """
        Scrapes the articles using nodejs script with article-extractor library
        """

        js_script_path = os.path.join(os.path.dirname(__file__), "extractus/scrape_articles.js")
        cmd = ["node", js_script_path, json.dumps(links)]
        json_result = subprocess.run(cmd, check=True, capture_output=True)

        result = json.loads(json_result.stdout.decode("utf-8"))

        articles = result["articles"]
        articles = [item for item in articles if item is not None]
        unprocessed = result["unprocessed"]
        
        return articles
