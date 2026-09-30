"""
job_matcher.py
Phase 2: AI-powered resume-to-job matching.

Uses the existing LLMClient (Groq/Gemini/Claude)
to compare a resume against a job posting.
"""

import json


def match_resume_to_job(
    llm_client,
    resume_text: str,
    job: dict,
) -> dict:
    """
    Compare one resume with one job and return structured
    matching results.
    """

    system_prompt = """
You are TrueFit, an AI resume and job matching assistant.

Your task is to compare a candidate's resume with a job posting.

Return ONLY valid JSON with exactly these fields:

{
  "match_score": 0,
  "matched_skills": [],
  "missing_skills": [],
  "reason": ""
}

Rules:
- match_score must be an integer from 0 to 100.
- matched_skills should contain skills clearly supported by the resume.
- missing_skills should contain important skills from the job that are
  not clearly present in the resume.
- reason should briefly explain the score.
- Do not invent skills or experience for the candidate.
- Do not penalize the candidate for skills that are not mentioned in the job.
- Focus on actual job requirements, skills, experience, and responsibilities.
"""

    job_title = job.get("title", "")
    company = job.get("company", "")
    description = job.get("description", "")

    prompt = f"""
CANDIDATE RESUME
----------------
{resume_text}

JOB POSTING
-----------
Title: {job_title}
Company: {company}

Description:
{description}

Compare the resume with this job posting and return the required JSON.
"""

    raw_response = llm_client.generate_json(
        system_prompt=system_prompt,
        prompt=prompt,
        max_tokens=1500,
    )

    try:
        result = json.loads(raw_response)

    except json.JSONDecodeError:
        # Try to recover JSON if the model added extra text.
        start = raw_response.find("{")
        end = raw_response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "Groq returned an invalid response. "
                f"Raw response: {raw_response}"
            )

        result = json.loads(raw_response[start:end + 1])

    # Make sure the score is safely converted to an integer.
    try:
        result["match_score"] = int(result.get("match_score", 0))
    except (ValueError, TypeError):
        result["match_score"] = 0

    result["match_score"] = max(
        0,
        min(100, result["match_score"])
    )

    result.setdefault("matched_skills", [])
    result.setdefault("missing_skills", [])
    result.setdefault("reason", "")

    return result
