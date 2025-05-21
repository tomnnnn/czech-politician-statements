import aiohttp
import sys
import asyncio

async def fetch(url, retries=0, ):
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.3"}
            async with session.get(url, headers=headers) as response:
                if response.status == 429 and retries < 3:
                    # retry after cooldown
                    print("Retrying")
                    await asyncio.sleep(3)
                    return await fetch(url, retries + 1)

                response.raise_for_status()
                encoding = response.charset or "utf-8"

                return await response.text(encoding=encoding, errors="replace")
        except asyncio.exceptions.TimeoutError as e:
            print(f"Request timed out for {url}: {e}", file=sys.stderr)
            return None
        except aiohttp.ClientError as e:
            print(f"Request failed for {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Request failed for {url}: {type(e)}", file=sys.stderr)
            return None
