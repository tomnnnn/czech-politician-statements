import dateparser

class StatementParser:
    """
    Given a statement div, parse the statement into a dictionary
    """

    date: str
    assessment: str
    statement: str
    author: str
    explanation: str
    explanation_brief: str
    tags: list
    origin: str
    party: str
    evidence_links: list = []
    selectors = {
        "citation": "div > div:nth-child(1) > div > div.ps-5 > cite",
        "statement": "div:nth-child(1) > div > div.ps-5 > a > blockquote > span:nth-child(2)",
        "author": "div > div:nth-child(1) > div > div.w-100px.min-w-100px > div.mt-2.text-center.w-100 > h3",
        "tags": "div > div.ps-5 > div",
        "assessment": "div > div:nth-child(2) > div.d-flex.align-items-center.mb-2 > span.text-primary.fs-5.text-uppercase.fw-600",
        "explanation": "div > div:nth-child(2) > div.accordion",
        "explanation_alt": "div > div:nth-child(2) > div.d-block",
        "date": "div:nth-child(1) > div > div:nth-child(1) > div > div.ps-5 > cite",
        "party": "div > div:nth-child(1) > div > div > div.px-2 > a > div > span"
    }

    def __init__(self, statement_div, with_evidence=True):
        self.statement_div = statement_div
        self.__parse(with_evidence)

    def get_dict(self):
        return {
            "label": self.assessment,
            "statement": self.statement,
            "author": self.author,
            "party": self.party,
            "explanation": self.explanation,
            "explanation_brief": self.explanation_brief,
            "date": self.date,
            "origin": self.origin,
            "tags": self.tags,
            "evidence_links": self.evidence_links
        }


    def __parse(self, include_evidence=False):
        self.__parse_citation()
        self.__parse_statement()
        self.__parse_assessment()
        self.__parse_explanation(with_evidence=include_evidence)
        self.__parse_tags()
        self.__parse_author()
        self.__parse_party()

    def __parse_citation(self):
        citation = self.statement_div.select(self.selectors["citation"])[0].get_text(strip=True)

        # last part of the citation is the date
        date = citation.split(",")[-1]
        # the rest is the origin
        origin = ', '.join(citation.split(",")[:-1])

        # parse the date from dd. m yyyy to YYY-mm-dd
        parsed_date = dateparser.parse(date, languages=["cs"])
        parsed_date = parsed_date.strftime("%Y-%m-%d") if parsed_date else "Unknown"

        if(parsed_date == "Unknown"):
            print(f"WARNING: Failed to parse date: {date}")

        self.date = parsed_date
        self.origin = origin

    def __parse_statement(self):
        statement = self.statement_div.select(self.selectors["statement"])[0].get_text(strip=True)
        # remove any words that contain "demagog" in them
        statement = ' '.join([word for word in statement.split() if "demagog" not in word.lower()])
        # notes by demagog are written as (pozn. Demagog), by removing words containing string demagog, 
        # we remove the closing parenthesis, so we need to put it back
        statement = statement.replace("pozn.", "pozn.)")

        self.statement = statement

    def __parse_assessment(self):
        assessment_div = self.statement_div.find( "div", {"data-sentry-component": "StatementAssessment"})
        assessment = assessment_div.findChildren("span", recursive=False)[1].get_text(strip=True)
        self.assessment = assessment

    def __parse_explanation(self, with_evidence=False):
        explanation_container = self.statement_div.select(self.selectors["explanation"])

        if explanation_container:
            explanation_brief = explanation_container[0].findChildren("div", recursive=False)[0].get_text(strip=True)
            explanation = ' '.join([p.get_text() for p in explanation_container[0].find_all("p")])
        else:
            explanation_container = self.statement_div.select(self.selectors["explanation_alt"])
            explanation_brief = ''
            explanation = ' '.join([p.get_text() for p in explanation_container[0].find_all("p")])

        if with_evidence:
            evidence_links = explanation_container[0].find_all("a")
            evidence_links = [{'url': link.get('href'), 'title': link.get_text()} for link in evidence_links]
            
            # remove links to expand the explanation and the link to the source
            evidence_links = [link for link in evidence_links if link['title'].lower() != "zobrazit celé odůvodnění" and link['title'].lower() != "trvalý odkaz"]

            self.evidence_links = evidence_links

        self.explanation = explanation
        self.explanation_brief = explanation_brief

    def __parse_tags(self):
        tags_container_selector = "div > div:nth-child(1) > div > div.ps-5 > div > div"

        tags_container = self.statement_div.select(tags_container_selector)
        tags = []
        if tags_container:
            tags = [span.get_text(strip=True) for span in tags_container[0].find_all("span")]
        self.tags = tags

    def __parse_author(self):
        author = self.statement_div.select(self.selectors["author"])[0].get_text(strip=True)
        self.author = author

    def __parse_party(self):
        party_span = self.statement_div.select(self.selectors["party"])
        party = party_span[0].get_text(strip=True) if party_span else ""
        self.party = party
