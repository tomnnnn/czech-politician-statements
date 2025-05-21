import asyncio
import sys
import aiohttp
from collections import defaultdict
from aiolimiter import AsyncLimiter
from fake_http_header import FakeHttpHeader
from urllib.parse import urlparse
from http.cookiejar import MozillaCookieJar
from requests.utils import dict_from_cookiejar

cookie_jar = MozillaCookieJar('cookies.txt')
COOKIES = dict_from_cookiejar(cookie_jar)

def find_by_id(id, data):
    for d in data:
        if d["id"] == id:
            return d
    return None

DOMAIN_RATE_LIMIT_MAP = defaultdict(lambda: AsyncLimiter(5, 1))

async def fetch(url, retries=3, timeout=30, backoff_factor=2, cookies={}):
    """
    Fetch data from the provided URL with retries in case of failure.

    Arguments:
        url (str): The URL to fetch data from.
        retries (int): The number of retry attempts (default is 3).
        timeout (int): The total timeout in seconds for the request (default is 30).
        backoff_factor (int): Factor for exponential backoff (default is 2).

    Returns:
        str or None: The fetched content or None if the request fails.
    """
    async def fetch_with_retry(url, retries_left):
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        domain = urlparse(url).netloc
        async with aiohttp.ClientSession(timeout=timeout_config, cookies=cookies) as session, DOMAIN_RATE_LIMIT_MAP[domain]:
            headers = FakeHttpHeader(domain_code="cz").as_header_dict()

            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 429 and retries_left > 0:
                        # Handle rate-limiting with retry and backoff
                        await asyncio.sleep(backoff_factor ** (3 - retries_left))  # Exponential backoff
                        return await fetch_with_retry(url, retries_left - 1)

                    response.raise_for_status()  # Raise an error for non-2xx status codes
                    encoding = response.charset or "utf-8"
                    return await response.text(encoding=encoding, errors="replace")

            except asyncio.exceptions.TimeoutError as e:
                print(f"Request timed out for {url}: {e}", file=sys.stderr)
            except aiohttp.ClientError as e:
                print(f"Request failed for {url}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Unexpected error for {url}: {e}", file=sys.stderr)

            return None

    # Start the fetch operation with retries
    return await fetch_with_retry(url, retries)
