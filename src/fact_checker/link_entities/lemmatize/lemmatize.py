"""
Advanced lemmatization functions capable of retaining gender in adjectives in compound nouns

@author: Hai Phong Nguyen
"""

from ufal.morphodita import Morpho, Tagger, Forms, TaggedLemmas, TokenRanges
import asyncio
import concurrent.futures
import os

# Load both dictionary and tagger
morpho = Morpho.load(os.path.join(os.path.dirname(__file__), "czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-220710.dict"))
tagger = Tagger.load(os.path.join(os.path.dirname(__file__), "czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-pdtc1.0-220710.tagger"))

def lemmatize(text: str) -> list:
    """
    Lemmatize the input text using the MorphoDita library.

    Args:
        text (str): The input text to be lemmatized.

    Returns:
        list: A list of clean lemmas extracted from the input text.
    """
    forms = Forms()
    tokens = TokenRanges()
    tokenizer = tagger.newTokenizer()
    tokenizer.setText(text)

    # Process the text and extract clean lemmas
    tagged_lemmas = TaggedLemmas()
    clean_results = []

    while tokenizer.nextSentence(forms, tokens):
        tagger.tag(forms, tagged_lemmas)
        
        for i in range(len(forms)):
            # Clean the lemma by removing technical annotations
            raw_lemma = tagged_lemmas[i].lemma
            clean_lemma = raw_lemma.split('-')[0].split('_')[0].split('^')[0]

            # Preserve case based on form's casing
            form = forms[i]  # Get the original form
            
            if form.isupper():  # All uppercase (e.g., STAN)
                clean_results.append(clean_lemma.upper())
            elif form[0].isupper() and form[1:].islower():  # Capitalized first letter (e.g., Praha)
                clean_results.append(clean_lemma.capitalize())
            else: 
                clean_results.append(clean_lemma)
            
    return clean_results

async def lemmatize_async(text: str) -> list:
    """
    Asynchronous wrapper for the lemmatize function.

    Args:
        text (str): The input text to be lemmatized.

    Returns:
        list: A list of clean lemmas extracted from the input text.
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, lemmatize, text)

if __name__ == "__main__":
    text = "Výsledky STEMu jasně ukazují preferenci pro strany ANO, ODS a Piráti."
    lemmas = lemmatize(text)
    print(lemmas)
