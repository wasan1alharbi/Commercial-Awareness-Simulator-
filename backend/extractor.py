"""
extractor.py: Entity Extraction Module
========================================
Parses news articles using Ollama to identify companies
and organisations mentioned. Uses low temperature for deterministic results. See report for comparative results
between different temperatures. I have tested 0.1 and 0.85.

Architecture adapted from the generative agent memory system described in:
Park, J.S. et al. (2023) 'Generative Agents: Interactive Simulacra of Human Behavior',
in Proceedings of UIST '23. ACM. doi:10.1145/3586183.3606763

The ai-town project (a16z-infra, 2023) provided a reference implementation
for structuring agent identity data and conversation orchestration.
Source: https://github.com/a16z-infra/ai-town (MIT License)
"""

import json
import re
import requests
import logging

logger = logging.getLogger(__name__)

# Ollama configuration - runs entirely locally, no external API calls
OLLAMA_BASE_URL = "http://localhost:11434"
EXTRACTION_MODEL = "llama3.2"  # Can be swapped for any Ollama-supported model
EXTRACTION_TEMPERATURE = 0.01   # Low temp for deterministic entity extraction
GENERATION_TEMPERATURE = 0.75  # Higher temp for creative agent responses


def check_ollama_health(model=EXTRACTION_MODEL):
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return {"healthy": False, "message": "Ollama is not responding"}

        models = resp.json().get("models", [])
        model_names = [m["name"].split(":")[0] for m in models]

        if model.split(":")[0] not in model_names:
            return {
                "healthy": False,
                "message": f"Model '{model}' not found. Available: {model_names}. "
                           f"Run: ollama pull {model}"
            }

        return {"healthy": True, "message": f"Ollama ready with {model}"}
    except requests.ConnectionError:
        return {
            "healthy": False,
            "message": "Cannot connect to Ollama. Make sure it is running (ollama serve)"
        }


def call_ollama(prompt, model=EXTRACTION_MODEL, temperature=EXTRACTION_TEMPERATURE):
    """Standardized wrapper for local Ollama API generation calls."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1024
                }
            },
            timeout=120  # LLM calls can take time on CPU
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama call failed: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def extract_companies(article_text):
    """Parses article text into structured JSON entities"""

    prompt = f"""Analyze this news article and extract ALL companies and organizations mentioned.

For each company, provide:
- name: The full company name
- slug: A URL-safe lowercase identifier (e.g. "goldman-sachs")
- sector: One of: Tech, Finance, Retail, Energy, Healthcare, Media, Automotive, Telecom, Consulting, Other

Return ONLY a JSON array. No explanation. Example format:
[{{"name": "Apple", "slug": "apple", "sector": "Tech"}}]

If no companies are found, return an empty array: []

Article:
{article_text[:3000]}

JSON:"""

    raw = call_ollama(prompt, temperature=EXTRACTION_TEMPERATURE)

    # Parse JSON from LLM response: to handle common formatting issues
    try:
        # Try to find JSON array in the response
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            companies = json.loads(match.group())
        else:
            companies = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse extraction response: {raw[:200]}")
        companies = []

    # Validate and clean each entry
    valid_sectors = {
        "Tech", "Finance", "Retail", "Energy", "Healthcare",
        "Media", "Automotive", "Telecom", "Consulting", "Other"
    }

    cleaned = []
    seen_slugs = set()
    for c in companies:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "").strip()
        slug = c.get("slug", "").strip().lower()
        sector = c.get("sector", "Other").strip()

        if not name or not slug:
            continue
        if slug in seen_slugs:
            continue
        if sector not in valid_sectors:
            sector = "Other"

        # Filter out non-business entities
        skip_keywords = ["government", "university", "ngo", "united nations"]
        if any(kw in name.lower() for kw in skip_keywords):
            continue

        seen_slugs.add(slug)
        cleaned.append({"name": name, "slug": slug, "sector": sector})

    logger.info(f"Extracted {len(cleaned)} companies from article")
    return cleaned


def get_article_summary(article_text):
    """Generates a concise 2-sentence business summary for when injecting article context into
    agent prompts"""
    prompt = f"""Summarize this news article in exactly 2 sentences.
Focus on the business and market implications.

Article:
{article_text[:2500]}

Summary:"""

    return call_ollama(prompt, temperature=0.2)
