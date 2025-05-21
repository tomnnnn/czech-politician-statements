"""
This script retrieves the introductory text (abstracts) of multiple Wikipedia articles using the Wikipedia API.

@author: Hai Phong Nguyen
"""

import requests
import simplemma

def get_wikipedia_abstracts(entities, lang='cs'):
    """
    Retrieves the introductory text (abstracts) of multiple Wikipedia articles in a single request
    and includes the entity type.

    :param entities: List of dictionaries containing 'name' and 'type' (e.g., [{'name': 'Albert Einstein', 'type': 'person'}, ...])
    :param lang: Language code for Wikipedia (default is 'cs' for Czech)
    :return: List of dictionaries with 'name', 'type', and 'abstract' for each entity
    """
    entity_names = [entity for entity in entities]
    titles = '|'.join(entity_names)  # Join entity names with pipe delimiter
    url = f'https://{lang}.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'titles': titles,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True
    }

    response = requests.get(url, params=params)
    data = response.json()

    abstracts = []
    pages = data.get('query', {}).get('pages', {})
    for page_id, page in pages.items():
        title = page.get('title')
        extract = page.get('extract', 'No abstract available')
        if not extract:
            data = search_and_retrieve(title, lang)
            extract = data['abstract'] if data else 'No abstract available'

        # Find the entity type from the input list
        entity = next((entity for entity in entities if entity == title), None)

        abstracts.append({
            'name': title,
            'abstract': extract
        })

    return abstracts


def search_and_retrieve(entity, lang):
    """
    Searches using /search API endpoint and retrieves the first result's abstract.
    """

    url = f'https://{lang}.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': entity,
        'format': 'json',
        'utf8': 1,
        'srlimit': 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    search_results = data.get('query', {}).get('search', [])
    if search_results:
        title = search_results[0]['title']
        return get_wikipedia_abstracts([{'name': title}], lang)[0]
    else:
        return None

# Example usage
entities = [
    'Petra Fialy'
]


abstracts = get_wikipedia_abstracts(entities)
for item in abstracts:
    print(f"Entity: {item}\nAbstract: {item['abstract']}\n")

