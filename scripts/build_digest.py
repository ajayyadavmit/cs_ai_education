"""Build a concise CS & AI Education article digest for GitHub Pages.

Search uses Google News RSS. Gemini is used for relevance assessment and
summaries of publicly accessible article text. The generated HTML contains
links and short summaries, not copied full articles.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai

# ============================================================
# USER SETTINGS: EDIT THESE VALUES
# ============================================================

# Gemini model. Change this only to a model available to your API key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# HOW MANY NEW ARTICLES TO SUMMARISE PER WORKFLOW RUN.
# Increase this number if you want more new articles processed each day.
# Example: change 10 to 20. Higher values use more Gemini requests.
MAX_NEW_ARTICLES_PER_RUN = 10

# Maximum number of articles retained and displayed on the HTML page.
# Increase this if you want a longer archive.
MAX_ARTICLES_ON_PAGE = 150

# Search window: recent items are prioritised; highly relevant older items
# can be included up to this many days old.
RECENT_DAYS = 30
OLDER_ARTICLE_LIMIT_DAYS = 365

# Maximum characters taken from an article page and sent to Gemini.
MAX_ARTICLE_TEXT_CHARS = 14000

# Pause between Gemini requests to reduce rate-limit pressure.
SECONDS_BETWEEN_GEMINI_CALLS = 2

# ------------------------------------------------------------
# SEMANTIC-STYLE SEARCH TERMS: ADD OR EDIT YOUR OWN TERMS HERE.
# ------------------------------------------------------------
# Google News RSS is a keyword/phrase search service, not a vector database.
# These carefully chosen phrases act as semantic search themes. Add terms
# that describe the ideas you want, even if publishers use different wording.
# They are used to build RSS queries and to score relevance.
#
# Examples you can add:
#   'AI-generated worksheets for primary school students'
#   'using large language models to teach Python'
#   'unplugged computational thinking activities'
#   'AI-powered formative assessment in K-12 classrooms'
#   'teaching data science to school children'
#   'interactive coding quizzes for beginners'
#
# Keep phrases reasonably short for better RSS search results.
SEMANTIC_SEARCH_TERMS = [
    "AI generated learning materials for school students",
    "AI tools for creating classroom quizzes and worksheets",
    "generative AI lesson plans for K-12 teachers",
    "teaching computer science with AI tools",
    "AI models for school education and student learning",
    "computational thinking activities for children",
    "coding games and programming activities for school students",
    "computer science teaching resources and classroom activities",
    "AI literacy curriculum for children and teenagers",
    "AI assisted assessment and feedback in schools",
    "learning analytics for K-12 education",
    "educational data mining and teaching analytics",
    "AI tutors and intelligent tutoring systems for school students",
    "generative AI for creating educational content",
    "computer science curriculum and programming pedagogy",
]

# Main subject areas. Add phrases here to tune local relevance scoring.
TOPIC_KEYWORDS = {
    "CS Education": [
        "computer science education", "k-12 computer science",
        "computing education", "computer science curriculum",
        "programming education", "coding education", "teaching python",
        "computer science teaching", "cs education",
    ],
    "AI Learning Materials": [
        "ai generated learning materials", "ai-generated worksheets",
        "ai lesson plan", "ai lesson plans", "ai generated quiz",
        "ai-generated quiz", "ai teaching materials", "educational content",
        "generative ai for teachers", "create learning materials",
        "ai classroom resources", "ai tutor", "intelligent tutoring",
    ],
    "Student Activities and Assessment": [
        "school students", "school children", "student activities",
        "classroom activities", "coding games", "coding quiz",
        "programming quiz", "formative assessment", "student assessment",
        "interactive learning", "unplugged activities", "worksheets",
    ],
    "Computational Thinking": [
        "computational thinking", "algorithmic thinking",
        "problem solving", "bebras", "computer science concepts",
    ],
    "AI in Education": [
        "generative ai", "artificial intelligence in education",
        "ai in education", "ai literacy", "large language models",
        "responsible ai", "ai-powered learning",
    ],
    "Teaching and Learning Analytics": [
        "learning analytics", "teaching analytics", "educational data mining",
        "student data", "learning data", "adaptive learning",
    ],
    "Digital Pedagogy": [
        "digital pedagogy", "teacher professional development",
        "educational technology", "digital learning", "personalised learning",
    ],
}

# Query combinations. These search both recent and older high-value material.
# You can add another query phrase or remove one if results are too broad.
BASE_SEARCH_QUERIES = [
    '"computer science education" (teaching OR curriculum OR classroom)',
    '"K-12 computer science" (teacher OR student OR resources)',
    '"AI in education" (school OR teacher OR classroom OR students)',
    '"generative AI" (lesson plans OR worksheets OR quizzes OR teaching)',
    '"computational thinking" (children OR school OR activities OR teaching)',
    '"learning analytics" (school OR classroom OR teaching)',
]

# ============================================================
# PATHS AND LOGGING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "articles.json"
HTML_FILE = ROOT / "cs-ai-education-digest.html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it under GitHub Settings > Secrets and variables > Actions."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; CSAIEducationDigest/1.0; +https://ajayyadav.com.np)"
})

TODAY = datetime.now(timezone.utc).date()


# ============================================================
# HELPERS
# ============================================================

def clean_url(url: str) -> str:
    """Remove common tracking parameters while retaining meaningful query data."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return url
        tracking_prefixes = ("utm_",)
        tracking_names = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith(tracking_prefixes)
            and key.lower() not in tracking_names
        ]
        return urlunparse((
            parsed.scheme.lower(), parsed.netloc.lower(),
            parsed.path.rstrip("/"), parsed.params,
            urlencode(query, doseq=True), ""
        ))
    except Exception:
        return url


def parse_entry_date(entry: Any) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc).date().isoformat()
        except (TypeError, ValueError):
            pass
    return ""


def age_in_days(date_text: str) -> int | None:
    if not date_text:
        return None
    try:
        return (TODAY - date.fromisoformat(date_text)).days
    except ValueError:
        return None


def classify_and_score(title: str, description: str = "") -> tuple[int, list[str]]:
    """Transparent phrase-based relevance score; tune keywords above."""
    text = f"{title} {description}".lower()
    score = 0
    categories: list[str] = []

    for category, phrases in TOPIC_KEYWORDS.items():
        hits = [phrase for phrase in phrases if phrase in text]
        if hits:
            categories.append(category)
            score += min(len(hits), 4) * 3

    # Extra weight for practical, school-facing materials.
    practical_signals = [
        "lesson plan", "worksheet", "quiz", "classroom activity",
        "learning materials", "teaching resources", "school students",
        "school children", "coding game", "student project",
        "teaching guide", "teacher guide", "assessment tool",
        "prompt library", "ai tool", "educational model",
    ]
    score += sum(5 for phrase in practical_signals if phrase in text)

    # Broad educational context helps distinguish education from general AI news.
    education_signals = [
        "teacher", "student", "school", "classroom", "curriculum",
        "learning", "teaching", "education", "pedagogy",
    ]
    score += min(sum(1 for phrase in education_signals if phrase in text), 5)

    return score, categories or ["Education Technology"]


def google_news_queries() -> list[str]:
    queries = list(BASE_SEARCH_QUERIES)
    queries.extend(f'"{term}"' for term in SEMANTIC_SEARCH_TERMS)
    return queries


# ============================================================
# ARTICLE DISCOVERY
# ============================================================

def discover_articles() -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    for query in google_news_queries():
        # Recent search: the last month.
        rss_url = (
            "https://news.google.com/rss/search?q="
            + requests.utils.quote(query + " when:30d")
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        logging.info("RSS search: %s", query)

        try:
            response = session.get(rss_url, timeout=25)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            logging.warning("RSS request failed: %s", exc)
            continue

        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            rss_link = (entry.get("link") or "").strip()
            if not title or not rss_link:
                continue

            description = BeautifulSoup(
                entry.get("summary", ""), "html.parser"
            ).get_text(" ", strip=True)
            score, categories = classify_and_score(title, description)
            published = parse_entry_date(entry)
            age = age_in_days(published)

            # Keep recent relevant results; older results must be more relevant.
            if age is not None:
                if age < 0 or age > OLDER_ARTICLE_LIMIT_DAYS:
                    continue
                if age > RECENT_DAYS and score < 12:
                    continue
            elif score < 15:
                continue

            # Avoid weakly related general technology news.
            if score < 6:
                continue

            publisher = ""
            source_obj = entry.get("source")
            if isinstance(source_obj, dict):
                publisher = (source_obj.get("title") or "").strip()

            key = clean_url(rss_link)
            item = {
                "title": title,
                "url": key,
                "rss_url": rss_link,
                "publisher": publisher,
                "author": (entry.get("author") or "").strip(),
                "published": published,
                "year": published[:4] if published else "",
                "description": description,
                "score": score,
                "categories": categories,
                "article_text": "",
                "summary": "",
                "summary_status": "pending",
            }

            if key not in found or score > found[key]["score"]:
                found[key] = item

    results = list(found.values())
    results.sort(
        key=lambda x: (x["score"], x.get("published", "")),
        reverse=True,
    )
    logging.info("Discovered %d candidate articles.", len(results))
    return results


# ============================================================
# FETCH PUBLISHER PAGE AND METADATA
# ============================================================

def fetch_article(article: dict[str, Any]) -> dict[str, Any]:
    """Resolve publisher URL, author, date and readable text where accessible."""
    try:
        response = session.get(article["rss_url"], timeout=30, allow_redirects=True)
        response.raise_for_status()
        final_url = clean_url(response.url)
        if final_url.startswith("http"):
            article["url"] = final_url

        downloaded = response.text
        metadata = trafilatura.extract_metadata(downloaded)
        if metadata:
            if metadata.author and not article.get("author"):
                article["author"] = metadata.author.strip()
            if metadata.date and not article.get("published"):
                try:
                    parsed_date = datetime.fromisoformat(
                        metadata.date.replace("Z", "+00:00")
                    ).date().isoformat()
                    article["published"] = parsed_date
                    article["year"] = parsed_date[:4]
                except ValueError:
                    pass
            if metadata.sitename and (
                not article.get("publisher")
                or "news.google.com" in article.get("publisher", "").lower()
            ):
                article["publisher"] = metadata.sitename.strip()

        soup = BeautifulSoup(downloaded, "html.parser")
        if not article.get("author"):
            for selector in [
                'meta[name="author"]',
                'meta[property="article:author"]',
                'meta[name="parsely-author"]',
            ]:
                tag = soup.select_one(selector)
                if tag and tag.get("content"):
                    article["author"] = tag["content"].strip()
                    break

        if not article.get("published"):
            for selector in [
                'meta[property="article:published_time"]',
                'meta[name="date"]',
                'meta[name="pubdate"]',
            ]:
                tag = soup.select_one(selector)
                if tag and tag.get("content"):
                    raw_date = tag["content"].strip()
                    try:
                        parsed_date = datetime.fromisoformat(
                            raw_date.replace("Z", "+00:00")
                        ).date().isoformat()
                        article["published"] = parsed_date
                        article["year"] = parsed_date[:4]
                        break
                    except ValueError:
                        pass

        if not article.get("publisher"):
            article["publisher"] = urlparse(article["url"]).netloc.replace("www.", "")

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            article["article_text"] = extracted[:MAX_ARTICLE_TEXT_CHARS]

    except Exception as exc:
        logging.warning("Could not retrieve %s: %s", article.get("url"), exc)

    if not article.get("publisher"):
        article["publisher"] = urlparse(article["url"]).netloc.replace("www.", "")
    if not article.get("year") and article.get("published"):
        article["year"] = article["published"][:4]
    if not article.get("author"):
        article["author"] = "Not listed"
    return article


# ============================================================
# GEMINI: SUMMARISE AND VERIFY OUTPUT
# ============================================================

def generate_gist(article: dict[str, Any]) -> str:
    source_text = (article.get("article_text") or "").strip()
    if len(source_text) < 250:
        source_text = (article.get("description") or "").strip()

    if len(source_text) < 80:
        raise ValueError(
            "Not enough article text or RSS description was accessible to create a reliable gist."
        )

    prompt = f"""
You are an education research assistant. Summarise this source for a reader
interested in school-level Computer Science and AI Education.

TITLE: {article.get('title', '')}
PUBLISHER: {article.get('publisher', '')}
AUTHOR: {article.get('author', 'Not listed')}
SOURCE TEXT (may be incomplete):
{source_text[:MAX_ARTICLE_TEXT_CHARS]}

Write a concise gist of 65-100 words. Prioritise practical value for teachers
and school learners. Explain what the resource, model, quiz, activity, tool,
research or teaching approach actually offers.

Use this format with these three labels:
Main idea: [central idea or resource]
Classroom use: [how a CS/AI teacher or school learner could use it]
Why it is useful: [the practical or research value]

Rules:
- Use only facts supported by the supplied source text.
- Do not invent the author, study results, features, age group or outcomes.
- If the text is only a short description, keep the claims cautious.
- If the source is not directly about education, state its limited relevance.
- Return plain text only, without markdown headings or introductory remarks.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty text response.")
    return text


def summarise_with_retry(article: dict[str, Any]) -> tuple[str, str]:
    last_error = "Unknown Gemini error"
    for attempt in range(3):
        try:
            gist = generate_gist(article)
            return gist, "ok"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "Gemini summary attempt %d/3 failed for '%s': %s",
                attempt + 1, article.get("title", ""), last_error
            )
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
    return "", last_error


# ============================================================
# DATABASE
# ============================================================

def load_database() -> dict[str, dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        return {}
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        database = {}
        for item in data:
            url = clean_url(item.get("url", ""))
            if not url:
                continue

            # Migrate failure messages saved by the previous script version.
            # Those messages are not real summaries and should be retried.
            old_summary = (item.get("summary") or "").strip().lower()
            old_failure_markers = (
                "summary generation failed",
                "the publisher did not provide enough accessible",
                "summary unavailable",
            )
            if any(marker in old_summary for marker in old_failure_markers):
                item["summary"] = ""
                item["summary_status"] = "error"

            # Older database entries may not have had these fields.
            item.setdefault("author", "Not listed")
            item.setdefault("publisher", "Publisher not listed")
            item.setdefault("year", (item.get("published", "") or "")[:4])
            item.setdefault("categories", [])
            item.setdefault("summary_status", "pending")
            database[url] = item

        return database
    except Exception as exc:
        logging.warning("Could not load article database: %s", exc)
        return {}


def save_database(articles: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# SIMPLE, RESPONSIVE HTML
# ============================================================

def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_card(article: dict[str, Any], index: int) -> str:
    title = esc(article.get("title", "Untitled article"))
    url = esc(article.get("url", "#"))
    publisher = esc(article.get("publisher", "Publisher not listed"))
    author = esc(article.get("author", "Not listed"))
    year = esc(article.get("year") or "Year not listed")
    gist = (article.get("summary") or "").strip()
    gist_html = "<br>".join(esc(line) for line in gist.splitlines())
    status = article.get("summary_status", "pending")
    status_note = ""
    if not gist:
        if status == "error":
            status_note = (
                '<p class="summary-note">Gist unavailable for this article. '
                'Check the workflow log for the Gemini or source-extraction error.</p>'
            )
        else:
            status_note = '<p class="summary-note">Summary is pending.</p>'

    topics = article.get("categories", [])
    topic_html = " · ".join(esc(topic) for topic in topics[:3])
    share_url = quote(article.get("url", ""), safe="")
    share_title = quote(article.get("title", ""), safe="")

    return f"""
    <article class="article" data-search="{esc((article.get('title','') + ' ' + article.get('publisher','') + ' ' + ' '.join(topics)).lower())}">
      <div class="meta"><span>{publisher}</span><span>{year}</span></div>
      <h2><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></h2>
      <p class="byline">Author: {author}</p>
      <p class="topics">{topic_html}</p>
      <details class="gist" {'open' if index == 0 and gist else ''}>
        <summary>+ Gist Summary</summary>
        <div class="gist-body">{gist_html}{status_note}</div>
      </details>
      <div class="actions">
        <a class="read" href="{url}" target="_blank" rel="noopener noreferrer">Read article ↗</a>
        <button type="button" class="share" data-url="{url}" data-title="{title}">Copy link</button>
        <a class="social" href="https://www.linkedin.com/sharing/share-offsite/?url={share_url}" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <a class="social" href="https://wa.me/?text={share_title}%20{share_url}" target="_blank" rel="noopener noreferrer">WhatsApp</a>
        <a class="social" href="https://www.facebook.com/sharer/sharer.php?u={share_url}" target="_blank" rel="noopener noreferrer">Facebook</a>
      </div>
    </article>
    """


def build_html(articles: list[dict[str, Any]]) -> None:
    cards = "\n".join(render_card(item, i) for i, item in enumerate(articles))
    if not cards:
        cards = '<p class="empty">No matching articles found yet. Try the next scheduled update.</p>'

    updated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Curated Computer Science and AI Education articles, teaching resources and learning analytics.">
<title>CS &amp; AI Education Digest</title>
<style>
:root {{ color-scheme: light; --bg:#f6f8f7; --paper:#fff; --ink:#202a27; --muted:#687570; --line:#e1e8e4; --green:#176b50; --soft:#edf6f1; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 system-ui,-apple-system,Segoe UI,Arial,sans-serif; }}
main {{ width:min(900px, calc(100% - 32px)); margin:0 auto; }}
header {{ padding:38px 0 22px; }}
h1 {{ margin:0 0 7px; font-size:clamp(27px,5vw,38px); line-height:1.2; letter-spacing:-.6px; }}
.intro {{ margin:0; color:var(--muted); max-width:700px; }}
.updated {{ margin:12px 0 0; color:var(--muted); font-size:12px; }}
.tools {{ position:sticky; top:0; z-index:2; padding:10px 0; background:rgba(246,248,247,.96); backdrop-filter:blur(8px); }}
#filter {{ width:100%; border:1px solid var(--line); border-radius:9px; padding:12px 14px; font:inherit; background:#fff; }}
.count {{ color:var(--muted); font-size:12px; margin:7px 2px 0; }}
.article {{ background:var(--paper); border:1px solid var(--line); border-radius:12px; padding:19px 20px; margin:13px 0; }}
.meta {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); font-size:12px; }}
h2 {{ font-size:19px; line-height:1.35; margin:9px 0 4px; }}
h2 a {{ color:var(--ink); text-decoration:none; }}
h2 a:hover {{ color:var(--green); text-decoration:underline; }}
.byline,.topics {{ margin:3px 0; color:var(--muted); font-size:13px; }}
.topics {{ color:var(--green); font-size:12px; }}
.gist {{ border-top:1px solid var(--line); margin-top:13px; padding-top:11px; }}
.gist summary {{ color:var(--green); font-weight:650; cursor:pointer; }}
.gist-body {{ padding:9px 0 2px; font-size:14px; white-space:normal; }}
.summary-note {{ color:#8a5a16; font-size:12px; margin:7px 0 0; }}
.actions {{ display:flex; flex-wrap:wrap; gap:9px; align-items:center; margin-top:12px; }}
.read {{ color:var(--green); font-weight:650; font-size:13px; text-decoration:none; }}
.share {{ border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink); padding:7px 10px; font-size:12px; cursor:pointer; }}
.share:hover {{ background:var(--soft); }}
.social {{ color:var(--muted); font-size:12px; text-decoration:none; }}
.social:hover {{ color:var(--green); text-decoration:underline; }}
.empty {{ background:#fff; border:1px solid var(--line); padding:20px; border-radius:10px; color:var(--muted); }}
footer {{ color:var(--muted); font-size:12px; padding:25px 0 35px; }}
#toast {{ position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:#202a27; color:white; padding:9px 14px; border-radius:8px; font-size:13px; display:none; z-index:5; }}
@media(max-width:520px) {{ main {{ width:calc(100% - 22px); }} header {{ padding-top:26px; }} .article {{ padding:15px; }} h2 {{ font-size:17px; }} .meta {{ font-size:11px; }} }}
</style>
</head>
<body>
<main>
<header>
  <h1>CS &amp; AI Education Digest</h1>
  <p class="intro">A concise collection of Computer Science and AI Education articles, school learning resources, classroom activities, quizzes, teaching models and learning analytics.</p>
  <p class="updated">Last updated: {esc(updated)} · {len(articles)} articles</p>
</header>
<div class="tools">
  <input id="filter" type="search" placeholder="Search title, publisher or topic..." aria-label="Filter articles">
  <p class="count" id="count" aria-live="polite">Showing {len(articles)} articles</p>
</div>
<section id="articles" aria-label="Articles">
{cards}
</section>
<footer>Summaries are AI-generated from accessible publisher text or RSS descriptions. Please visit the original source for full context. Articles and resources remain the property of their respective publishers.</footer>
</main>
<div id="toast" role="status" aria-live="polite"></div>
<script>
const filterBox = document.getElementById('filter');
const cards = [...document.querySelectorAll('.article')];
filterBox.addEventListener('input', () => {{
  const q = filterBox.value.trim().toLowerCase();
  let visible = 0;
  cards.forEach(card => {{
    const match = card.dataset.search.includes(q) || card.innerText.toLowerCase().includes(q);
    card.hidden = !match;
    if (match) visible++;
  }});
  document.getElementById('count').textContent = `Showing ${{visible}} of ${{cards.length}} articles`;
}});
function toast(message) {{
  const el = document.getElementById('toast');
  el.textContent = message;
  el.style.display = 'block';
  window.setTimeout(() => el.style.display = 'none', 2300);
}}
document.querySelectorAll('.share').forEach(button => {{
  button.addEventListener('click', async () => {{
    const url = new URL(button.dataset.url, window.location.href).href;
    const title = button.dataset.title || 'Article';
    try {{
      await navigator.clipboard.writeText(`${{title}}\n${{url}}`);
      toast('Article link copied');
    }} catch (e) {{
      window.prompt('Copy this article link:', url);
    }}
  }});
}});
</script>
</body>
</html>'''
    HTML_FILE.write_text(page, encoding="utf-8")
    logging.info("HTML page generated: %s", HTML_FILE)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main() -> None:
    database = load_database()
    discovered = discover_articles()

    # Merge new discoveries while retaining existing summaries.
    for item in discovered:
        key = clean_url(item["url"])
        if key in database:
            old = database[key]
            old["score"] = max(int(old.get("score", 0)), item["score"])
            old["categories"] = sorted(set(
                old.get("categories", []) + item.get("categories", [])
            ))
            if not old.get("publisher"):
                old["publisher"] = item.get("publisher", "")
            if not old.get("author") or old.get("author") == "Not listed":
                old["author"] = item.get("author", "Not listed")
            if not old.get("published") and item.get("published"):
                old["published"] = item["published"]
                old["year"] = item.get("year", "")
        else:
            item["first_seen"] = TODAY.isoformat()
            database[key] = item

    # Keep only the configured archive window.
    retained: list[dict[str, Any]] = []
    for item in database.values():
        age = age_in_days(item.get("published", ""))
        if age is not None and age > OLDER_ARTICLE_LIMIT_DAYS:
            continue
        retained.append(item)

    # Highest-scoring unsummarised articles are processed first.
    pending = [a for a in retained if not a.get("summary")]
    pending.sort(
        key=lambda a: (int(a.get("score", 0)), a.get("published", "")),
        reverse=True,
    )

    processed = 0
    for article in pending[:MAX_NEW_ARTICLES_PER_RUN]:
        logging.info("Preparing article: %s", article.get("title", ""))
        fetch_article(article)
        gist, status = summarise_with_retry(article)
        article["summary"] = gist
        article["summary_status"] = "ok" if status == "ok" else "error"
        article["summary_error"] = "" if status == "ok" else status
        article["summarised_at"] = datetime.now(timezone.utc).isoformat()
        processed += 1
        logging.info(
            "Summary status for '%s': %s",
            article.get("title", ""), article["summary_status"]
        )
        time.sleep(SECONDS_BETWEEN_GEMINI_CALLS)

    # Sort newest first, then relevance. Unknown dates appear after dated items.
    retained.sort(
        key=lambda a: (
            bool(a.get("published")),
            a.get("published", ""),
            int(a.get("score", 0)),
        ),
        reverse=True,
    )
    retained = retained[:MAX_ARTICLES_ON_PAGE]

    save_database(retained)
    build_html(retained)

    ok_count = sum(1 for a in retained if a.get("summary_status") == "ok")
    error_count = sum(1 for a in retained if a.get("summary_status") == "error")
    logging.info(
        "Finished: %d articles displayed; %d processed this run; %d summaries available; %d summary errors.",
        len(retained), processed, ok_count, error_count,
    )


if __name__ == "__main__":
    main()
