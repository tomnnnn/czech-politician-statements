"""
This script provides function to detect named entities in a given text using the Nametag3 API.

@author: Hai Phong Nguyen

"""

import aiohttp
from .lemmatize.lemmatize import lemmatize_async
from threading import Semaphore
import asyncio

SEM = Semaphore(10)
ENDPOINT = 'http://lindat.mff.cuni.cz/services/nametag/api/recognize'

def parse_vertical_response(vertical_response):
    # split the response into lines, throw away the last empty \n
    rows = vertical_response.split('\n')[:-1]

    entities = []
    for row in rows:
        cols = row.split('\t')
        if cols[1] in ["PER", "ORG", "P", "io", "or", "if", "i"]:
            indices = cols[0].split(',')
            start = int(indices[0]) - 1
            end = int(indices[-1]) - 1

            entities.append({
                "name": cols[2],
                "start": start,
                "end": end,
            })

    return entities


async def detect_entities(text: str, lang: str, do_lemmatize: bool = True):
    # Prepare data and files
    data = {
        'model': lang,
        'data': text,
        'output': 'vertical',
    }
    
    SEM.acquire()
    await asyncio.sleep(1)

    async with aiohttp.ClientSession() as session:
        async with session.post(ENDPOINT, data=data) as response:
            # Handle the response
            if response.status == 200:
                response_json = await response.json()
                entities = parse_vertical_response(response_json['result'])
            else:
                SEM.release()
                raise Exception(f"Request failed with status {response.status}")

    SEM.release()

    # Lemmatize the entities if required
    if do_lemmatize:
        # lemmatize the whole text to provide context
        lemmatized_words = await lemmatize_async(text)
        lemmatized_entities = [ " ".join(lemmatized_words[entity['start']:entity['end'] + 1]) for entity in entities ]
        result = lemmatized_entities
    else:
        result = [entity['name'] for entity in entities]

    return result

    

async def detect_entities_batch(texts: list, lang: str, doLemmatize: bool = True):
    coroutines = [detect_entities(text, lang, doLemmatize) for text in texts]
    results = await asyncio.gather(*coroutines)
    return results

if __name__ == "__main__":
    import asyncio
    texts = [
        "Miloš Zeman s Petrem Fialou přijeli do Hyundaye",
        "Rada OSN pristoupila na zákonou změnu v Zákonu o ochraně životního prostředí",
        "Barack Obama byl prezidentem USA",
        "Výsledky STEMu jasně ukazují preferenci pro strany ANO, ODS a Piráti."
    ]
    lang = "cs"
    
    results = asyncio.run(detect_entities_batch(texts, lang, doLemmatize=True))
    print(results)
