"""
job_search.py
Phase 2: Automated job search using the public Himalayas Jobs API.

No API key or signup required.
"""

import requests

HIMALAYAS_BASE_URL = "https://himalayas.app/jobs/api"


def check_api() -> bool:
    """Check whether the Himalayas API is reachable."""

    response = requests.get(
        HIMALAYAS_BASE_URL,
        params={"limit": 1},
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    return "jobs" in data


def search_jobs(
    query: str = "",
    country: str = "",
    seniority: str = "",
    employment_type: str = "",
    page: int = 1,
    limit: int = 20,
) -> list:
    """
    Search remote jobs from Himalayas.

    Returns a clean list of dictionaries that TrueFit can
    later compare against the user's resume.
    """

    params = {
        "limit": min(limit, 20),
        "page": page,
    }

    if query.strip():
        params["q"] = query.strip()

    if country.strip():
        params["country"] = country.strip()

    if seniority.strip():
        params["seniority"] = seniority.strip()

    if employment_type.strip():
        params["employment_type"] = employment_type.strip()

    response = requests.get(
        f"{HIMALAYAS_BASE_URL}/search",
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    jobs = []

    for item in data.get("jobs", []):
        jobs.append(
            {
                "id": item.get("guid", ""),
                "title": item.get("title", "Untitled role"),
                "company": item.get("companyName", "Unknown company"),
                "description": item.get("description", ""),
                "excerpt": item.get("excerpt", ""),
                "location": ", ".join(
                    item.get("locationRestrictions", [])
                ),
                "employment_type": item.get(
                    "employmentType", ""
                ),
                "seniority": item.get("seniority", ""),
                "category": item.get("categories", []),
                "salary_min": item.get("minSalary"),
                "salary_max": item.get("maxSalary"),
                "currency": item.get("currency", ""),
                "posted_at": item.get("pubDate", ""),
                "application_link": item.get(
                    "applicationLink", ""
                ),
            }
        )

    return jobs