"""
This script provides functions to resolve abstracts for given named entities using DBpedia.

@author: Hai Phong Nguyen
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET

# --- DBpedia endpoints ---
SPARQL_ENDPOINT = "http://dbpedia.org/sparql"
LOOKUP_URL = "https://lookup.dbpedia.org/api/search/KeywordSearch"

# --- Rate limiting ---
LIMITER = asyncio.Semaphore(99)  # Limit to 20 concurrent requests

# --- SPARQL query template ---
query_template = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?label ?abstract WHERE {{
    VALUES ?entity {{ {entities} }}
    ?entity dbo:abstract ?abstract.
    ?entity rdfs:label ?label.
    FILTER (lang(?abstract) = '{lang}')
    FILTER (lang(?label) = '{lang}')
}}
"""

# --- Format SPARQL query ---
def format_query(entities, lang):
    values = " ".join(f"<http://dbpedia.org/resource/{entity}>" for entity in entities)
    return query_template.format(entities=values, lang=lang)


# --- Search entity by exact label ---
async def search_entities_by_labels(session, labels, lang):
    """
    Search for entities by their labels using SPARQL. Gives exactly one entity for each label.
    """
    # Create a VALUES clause to match multiple labels
    values_clause = " ".join([f'"{label}"@{lang}' for label in labels])
    
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?label (MIN(?entity) AS ?entity) WHERE {{
      ?entity rdfs:label ?label .
      VALUES ?label {{ {values_clause} }}
    }}
    GROUP BY ?label
    """
    params = {
        'query': query,
        'format': 'json'
    }
    async with session.get(SPARQL_ENDPOINT, params=params) as response, LIMITER:
        await asyncio.sleep(1)  # Rate limiting
        if response.status == 200:
            data = await response.json()
            bindings = data['results']['bindings']
            if bindings and bindings[0]:
                entities = [binding['entity']['value'].split('/')[-1] for binding in bindings]
                return entities
            else:
                return []
        else:
            print(f"SPARQL search error for labels {labels}: {response.status}")
            return []


# --- Lookup entity with fallback (XML parsing) ---
async def lookup_entity(session, keyword):
    headers = {
        "Accept": "application/xml"
    }
    params = {
        "query": keyword,
        "maxResults": 1
    }
    async with session.get(LOOKUP_URL, headers=headers, params=params) as response, LIMITER:
        await asyncio.sleep(1)  # Rate limiting
        if response.status == 200:
            text = await response.text()
            root = ET.fromstring(text)

            result = root.find('.//Result')
            if result is not None:
                uri = result.find('URI')
                if uri is not None:
                    entity = uri.text.split('/')[-1]
                    return entity
            return None
        else:
            print(f"Lookup Error for {keyword}: {response.status}")
            return None

# --- Smart resolver: tries SPARQL first, then fallback ---
async def resolve_entities(session, entities, lang):
    # Try to resolve entities using SPARQL first
    resolved_entities = await search_entities_by_labels(session, entities, lang)
    
    # if not resolved_entities:
    #     # If SPARQL fails, fallback to lookup
    #     tasks = [lookup_entity(session, entity) for entity in entities]
    #     resolved_entities = await asyncio.gather(*tasks)
    #     resolved_entities = [entity for entity in resolved_entities if entity is not None]
    #
    return resolved_entities

# --- Fetch abstracts ---
async def fetch_abstracts(session, entities, lang) -> list[dict[str,str]]:
    if not entities:
        return []

    query = format_query(entities, lang)
    async with session.get(SPARQL_ENDPOINT, params={'query': query, 'format': 'json'}) as response, LIMITER:
        await asyncio.sleep(1)  # Rate limiting
        if response.status == 200:
            data = await response.json()
            results = data['results']['bindings']

            return [
                {
                    "title": item["label"]["value"],
                    "text": item["abstract"]["value"] 
                }
                for item in results
            ]
        else:
            print(f"SPARQL Error: {response.status}")
            return []

async def get_abstracts(entities, lang) -> list[dict[str,str]]:
    async with aiohttp.ClientSession() as session:
        resolved_entities = await resolve_entities(session, entities, lang)

        if not resolved_entities:
            return []

        results = await fetch_abstracts(session, resolved_entities, lang)
        return results


# --- Main pipeline ---
async def main(raw_entities, lang="en"):
        results = await get_abstracts(raw_entities, lang)

        # Print results
        for result in results:
            print(result)
            print("")

if __name__ == "__main__":
    # Example rough input
    entities = [
        "Petr Fiala"
    ]
    asyncio.run(main(entities, "cs"))

