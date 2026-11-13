
import os
import re
import json
import time
import html
import logging
import feedparser
import requests
import trafilatura

from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, urlparse, urlunparse
from bs4 import BeautifulSoup
from google import genai

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "articles.json"
OUTPUT_FILE = ROOT / "cs-ai-education-digest.html"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

MAX_NEW_SUMMARIES = 6
MAX_ARTICLES = 120
MAX_OLD_ARTICLE_DAYS = 365

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=GEMINI_API_KEY)

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; "
        "CS-AI-Education-Digest/1.0)"
    )
})

# =========================================================
# SEARCH QUERIES
# =========================================================

TODAY = datetime.now(timezone.utc).date()
ONE_YEAR_AGO = TODAY - timedelta(days=365)

SEARCH_QUERIES = [
    # Recent articles
    '("computer science education" OR '
    '"K-12 computer science" OR '
    '"programming education") '
    '(teacher OR curriculum OR classroom) when:30d',

    '("AI in education" OR "generative AI" OR '
    '"AI literacy") '
    '(teacher OR classroom OR curriculum OR school) when:30d',

    '("computational thinking" OR "coding education") '
    '(teaching OR assessment OR curriculum) when:30d',

    '("learning analytics" OR "teaching analytics" OR '
    '"educational data mining") '
    '(teacher OR classroom OR school) when:30d',

    # Older but potentially important articles
    f'("computer science education" OR '
    f'"computational thinking" OR "AI literacy") '
    f'(teaching OR curriculum OR pedagogy OR assessment) '
    f'after:{ONE_YEAR_AGO.isoformat()}',

    f'("learning analytics" OR "educational data mining" OR '
    f'"AI in education") '
    f'(teaching OR school OR classroom) '
    f'after:{ONE_YEAR_AGO.isoformat()}'
]

# =========================================================
# TOPIC CLASSIFICATION AND RELEVANCE
# =========================================================

TOPICS = {
    "Computer Science Education": [
        "computer science education",
        "k-12 computer science",
        "cs education",
        "programming education",
        "coding education",
        "computer science curriculum"
    ],

    "AI in Education": [
        "generative ai",
        "ai in education",
        "artificial intelligence in education",
        "ai literacy",
        "ai tutor",
        "ai-assisted teaching"
    ],

    "Computational Thinking": [
        "computational thinking",
        "problem solving",
        "programming pedagogy",
        "coding curriculum"
    ],

    "Teaching & Learning Analytics": [
        "learning analytics",
        "teaching analytics",
        "educational data mining",
        "student analytics",
        "learning data"
    ],

    "Digital Pedagogy": [
        "digital pedagogy",
        "digital learning",
        "educational technology",
        "personalised learning",
        "adaptive learning",
        "teacher professional development"
    ]
}

HIGH_VALUE_PHRASES = [
    "computer science education",
    "computational thinking",
    "ai literacy",
    "generative ai in education",
    "programming education",
    "k-12 computer science",
    "learning analytics",
    "educational data mining",
    "computer science curriculum"
]

SUPPORTING_PHRASES = [
    "teacher",
    "classroom",
    "curriculum",
    "assessment",
    "pedagogy",
    "school",
    "teaching",
    "coding",
    "student",
    "instruction",
    "educational technology",
    "digital learning"
]


def analyse_relevance(title, description=""):
    text = f"{title} {description}".lower()

    score = 0
    categories = []

    for category, phrases in TOPICS.items():
        matched = [
            phrase for phrase in phrases
            if phrase in text
        ]

        if matched:
            categories.append(category)
            score += len(matched) * 2

    for phrase in HIGH_VALUE_PHRASES:
        if phrase in text:
            score += 4

    for phrase in SUPPORTING_PHRASES:
        if phrase in text:
            score += 1

    if not categories:
        categories = ["Education Technology"]

    return score, categories


# =========================================================
# URL AND DATE HELPERS
# =========================================================

def clean_url(url):
    """Remove tracking parameters and fragments."""

    try:
        parsed = urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            return url

        return urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
            ""
        ))

    except Exception:
        return url


def get_publication_date(entry):
    """Read publication date from RSS metadata."""

    published = entry.get("published_parsed")

    if published:
        return datetime(
            *published[:6],
            tzinfo=timezone.utc
        ).date().isoformat()

    return ""


def get_source_name(entry, final_url):
    source = entry.get("source", {})

    if isinstance(source, dict):
        name = source.get("title", "")
        if name:
            return name

    title = entry.get("title", "")

    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()

    return urlparse(final_url).netloc.replace("www.", "")


# =========================================================
# RSS SEARCH
# =========================================================

def search_articles():
    collected = {}

    for query in SEARCH_QUERIES:

        rss_url = (
            "https://news.google.com/rss/search?q="
            + quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        )

        logging.info("Searching: %s", query)

        try:
            response = session.get(
                rss_url,
                timeout=30
            )
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            for entry in feed.entries:

                title = entry.get("title", "").strip()
                rss_link = entry.get("link", "").strip()

                if not title or not rss_link:
                    continue

                description = BeautifulSoup(
                    entry.get("summary", ""),
                    "html.parser"
                ).get_text(" ", strip=True)

                score, categories = analyse_relevance(
                    title,
                    description
                )

                published = get_publication_date(entry)

                if published:
                    pub_date = datetime.fromisoformat(
                        published
                    ).date()

                    age = (TODAY - pub_date).days

                    # Recent articles need reasonable relevance.
                    is_recent = (
                        0 <= age <= 30 and score >= 4
                    )

                    # Older articles must be highly relevant.
                    is_evergreen = (
                        30 < age <= MAX_OLD_ARTICLE_DAYS
                        and score >= 8
                    )

                    if not (is_recent or is_evergreen):
                        continue

                else:
                    # Unknown dates are only retained
                    # when the relevance is particularly high.
                    if score < 12:
                        continue

                url = clean_url(rss_link)

                if url not in collected:
                    collected[url] = {
                        "title": title,
                        "url": url,
                        "rss_url": rss_link,
                        "description": description,
                        "published": published,
                        "score": score,
                        "categories": categories,
                        "source": "",
                        "article_text": ""
                    }

        except Exception as error:
            logging.warning(
                "RSS search failed: %s",
                error
            )

    results = list(collected.values())

    results.sort(
        key=lambda item: (
            item["score"],
            item["published"]
        ),
        reverse=True
    )

    logging.info(
        "Collected %s relevant articles.",
        len(results)
    )

    return results


# =========================================================
# ARTICLE RETRIEVAL
# =========================================================

def retrieve_article(article):
    """Resolve the publisher URL and extract accessible text."""

    url = article["rss_url"]

    try:
        response = session.get(
            url,
            timeout=25,
            allow_redirects=True
        )

        response.raise_for_status()

        final_url = clean_url(response.url)

        if final_url:
            article["url"] = final_url

        article["source"] = get_source_name(
            {
                "title": article["title"],
                "source": {}
            },
            final_url
        )

        extracted = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        if extracted:
            article["article_text"] = extracted[:12000]

        else:
            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            for element in soup([
                "script",
                "style",
                "nav",
                "footer",
                "header"
            ]):
                element.decompose()

            article["article_text"] = (
                soup.get_text(" ", strip=True)[:12000]
            )

    except Exception as error:
        logging.warning(
            "Could not extract article: %s",
            error
        )

    if not article.get("source"):
        article["source"] = urlparse(
            article["url"]
        ).netloc.replace("www.", "")

    return article


# =========================================================
# GEMINI SUMMARY
# =========================================================

def generate_summary(article):

    content = (
        article.get("article_text")
        or article.get("description")
    )

    if not content or len(content.strip()) < 80:
        return (
            "The publisher did not provide enough accessible "
            "article text for a reliable summary."
        ), "Insufficient source text"

    if article.get("article_text"):
        basis = "Full article text extracted"
    else:
        basis = "Publisher RSS description only"

    prompt = f"""
You are an academic research assistant specialising in:

- Computer Science Education
- AI in Education
- Computational Thinking
- Digital Pedagogy
- Teaching and Learning Analytics

Summarise the following published article.

ARTICLE TITLE:
{article['title']}

PUBLISHER:
{article['source']}

SOURCE TEXT:
{content[:12000]}

Write a concise gist summary in approximately 80-120 words.

Use exactly these three labelled sections:

Core idea:
Explain the central argument, finding or development.

Teaching connection:
Explain how it may relate to Computer Science teaching,
teacher professional development or educational practice.

Why it matters:
Explain its potential significance for educators,
curriculum developers or researchers.

IMPORTANT:
- Use only information supported by the supplied text.
- Do not invent statistics, findings or research methods.
- If the text is incomplete, acknowledge the limitation.
- Do not pretend to have read inaccessible sections.
- Use clear professional English.
"""

    for attempt in range(3):

        try:
            result = client.interactions.create(
                model=GEMINI_MODEL,
                input=prompt
            )

            summary = (result.output_text or "").strip()

            if summary:
                return summary, basis

        except Exception as error:
            logging.warning(
                "Gemini attempt %s failed: %s",
                attempt + 1,
                error
            )

            time.sleep(3 * (attempt + 1))

    return (
        "Summary generation failed. Please visit the "
        "original publisher article."
    ), "Summary unavailable"


# =========================================================
# LOAD AND SAVE EXISTING ARTICLES
# =========================================================

def load_existing():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        return {}

    try:
        with DATA_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            articles = json.load(file)

        return {
            item["url"]: item
            for item in articles
            if item.get("url")
        }

    except Exception as error:
        logging.warning(
            "Could not read existing database: %s",
            error
        )
        return {}


def save_articles(articles):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            indent=2,
            ensure_ascii=False
        )


# =========================================================
# HTML GENERATION
# =========================================================

def safe(value):
    return html.escape(str(value or ""))


def render_article(article):

    title = safe(article.get("title"))
    source = safe(article.get("source"))
    url = safe(article.get("url"))
    published = safe(
        article.get("published") or "Date not provided"
    )

    categories = article.get("categories", [])

    category_html = " ".join(
        f'<span class="tag">{safe(category)}</span>'
        for category in categories
    )

    summary = safe(article.get("summary")).replace(
        "\n",
        "<br>"
    )

    basis = safe(article.get("summary_basis"))

    score = int(article.get("score", 0))

    if score >= 12:
        relevance_label = "Highly relevant"
    else:
        relevance_label = "Relevant"

    return f"""
    <article class="article-card"
             data-title="{title.lower()}"
             data-category="{safe(' '.join(categories).lower())}">

        <div class="card-top">
            <span class="relevance">{relevance_label}</span>
            <time>{published}</time>
        </div>

        <h2>{title}</h2>

        <p class="publisher">
            {source}
        </p>

        <div class="tags">
            {category_html}
        </div>

        <details class="summary-box">
            <summary>+ Gist Summary</summary>

            <div class="summary-content">
                {summary}
                <p class="basis">
                    Summary basis: {basis}
                </p>
            </div>
        </details>

        <a class="article-link"
           href="{url}"
           target="_blank"
           rel="noopener noreferrer">
            Open original article ↗
        </a>

    </article>
    """


def generate_html(articles):

    updated = datetime.now(
        timezone.utc
    ).strftime("%d %B %Y, %H:%M UTC")

    cards = "\n".join(
        render_article(article)
        for article in articles
    )

    if not cards:
        cards = """
        <p class="empty">
            No matching articles were found.
            Please check again after the next update.
        </p>
        """

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>CS & AI Education Digest | Ajay Yadav</title>

<meta name="description"
content="A curated daily digest of Computer Science Education,
AI in Education, Computational Thinking and Learning Analytics.">

<style>
:root {{
    --bg: #f7f8fa;
    --surface: #ffffff;
    --text: #20252d;
    --muted: #687385;
    --border: #e4e8ee;
    --accent: #176b52;
    --accent-light: #e8f4ef;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.65;
}}

.container {{
    width: min(100% - 32px, 900px);
    margin: auto;
}}

.hero {{
    padding: 55px 0 35px;
}}

.eyebrow {{
    color: var(--accent);
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

h1 {{
    font-size: clamp(30px, 5vw, 44px);
    line-height: 1.15;
    margin: 12px 0;
}}

.subtitle {{
    color: var(--muted);
    max-width: 680px;
}}

.updated {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 18px;
}}

.search {{
    width: 100%;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    font-size: 15px;
    margin: 10px 0 25px;
}}

.article-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 18px;
}}

.card-top {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
}}

.relevance {{
    color: var(--accent);
    background: var(--accent-light);
    border-radius: 20px;
    padding: 4px 10px;
    font-weight: bold;
}}

h2 {{
    font-size: 21px;
    line-height: 1.35;
    margin: 16px 0 5px;
}}

.publisher {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 5px;
}}

.tags {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 13px 0;
}}

.tag {{
    background: #f0f2f5;
    color: #465264;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 11px;
}}

.summary-box {{
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 13px 0;
    margin: 18px 0;
}}

.summary-box summary {{
    cursor: pointer;
    color: var(--accent);
    font-weight: bold;
}}

.summary-content {{
    padding-top: 13px;
    color: #394454;
    font-size: 14px;
}}

.basis {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 15px;
}}

.article-link {{
    color: var(--accent);
    text-decoration: none;
    font-size: 14px;
    font-weight: bold;
}}

.article-link:hover {{
    text-decoration: underline;
}}

.empty {{
    padding: 30px;
    background: white;
    border-radius: 10px;
    color: var(--muted);
}}

footer {{
    padding: 35px 0;
    color: var(--muted);
    font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 35px;
}}

@media (max-width: 600px) {{
    .hero {{
        padding-top: 35px;
    }}

    .article-card {{
        padding: 18px;
    }}

    h2 {{
        font-size: 19px;
    }}
}}
</style>
</head>

<body>

<header class="container hero">

    <div class="eyebrow">
        Research and Professional Learning
    </div>

    <h1>CS & AI Education Digest</h1>

    <p class="subtitle">
        A curated collection of articles, research insights
        and practical developments in Computer Science
        Education, Artificial Intelligence, Computational
        Thinking and Teaching Analytics.
    </p>

    <p class="updated">
        Last updated: {safe(updated)}
    </p>

</header>

<main class="container">

    <input
        type="search"
        class="search"
        id="articleSearch"
        placeholder="Search articles, topics or publishers..."
        aria-label="Search articles"
    >

    <div id="articleList">
        {cards}
    </div>

</main>

<footer>
    <div class="container">
        Articles remain the property of their respective
        publishers. This digest provides links to original
        sources and AI-generated summaries for educational
        discovery. Always consult the original article for
        complete context.
        <br><br>
        Curated with Google Gemini and GitHub Actions.
    </div>
</footer>

<script>
const searchBox = document.getElementById("articleSearch");

searchBox.addEventListener("input", function () {{
    const query = this.value.toLowerCase();

    document.querySelectorAll(".article-card").forEach(card => {{
        const text = card.innerText.toLowerCase();
        card.style.display = text.includes(query) ? "" : "none";
    }});
}});
</script>

</body>
</html>
"""

    OUTPUT_FILE.write_text(
        page,
        encoding="utf-8"
    )

    logging.info("HTML digest generated.")


# =========================================================
# MAIN
# =========================================================

def main():

    existing = load_existing()

    discovered = search_articles()

    today_string = TODAY.isoformat()

    # Merge newly discovered articles.
    for item in discovered:

        url = item["url"]

        if url in existing:
            old = existing[url]

            # Keep the previous summary.
            old["score"] = max(
                old.get("score", 0),
                item["score"]
            )

            old["categories"] = list(set(
                old.get("categories", [])
                + item["categories"]
            ))

            if not old.get("source"):
                old["source"] = item["source"]

        else:
            item["first_seen"] = today_string
            item["summary"] = ""
            item["summary_basis"] = ""
            existing[url] = item

    # Remove articles outside the retention window.
    retained = []

    for item in existing.values():

        published = item.get("published")

        if published:
            age = (
                TODAY
                - datetime.fromisoformat(published).date()
            ).days

            if age > MAX_OLD_ARTICLE_DAYS:
                continue

        retained.append(item)

    # Prioritise new summaries.
    retained.sort(
        key=lambda item: (
            bool(item.get("summary")),
            item.get("score", 0),
            item.get("published", "")
        )
    )

    summary_count = 0

    for item in retained:

        if item.get("summary"):
            continue

        if summary_count >= MAX_NEW_SUMMARIES:
            break

        logging.info(
            "Processing: %s",
            item["title"]
        )

        item = retrieve_article(item)

        summary, basis = generate_summary(item)

        item["summary"] = summary
        item["summary_basis"] = basis
        item["last_summarised"] = datetime.now(
            timezone.utc
        ).isoformat()

        summary_count += 1

        time.sleep(2)

    # Sort for display.
    retained.sort(
        key=lambda item: (
            item.get("published", ""),
            item.get("score", 0)
        ),
        reverse=True
    )

    retained = retained[:MAX_ARTICLES]

    save_articles(retained)
    generate_html(retained)

    logging.info(
        "Completed. Total articles: %s",
        len(retained)
    )

    logging.info(
        "New summaries generated: %s",
        summary_count
    )


if __name__ == "__main__":
    main()
