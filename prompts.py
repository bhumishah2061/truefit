"""
prompts.py
Centralized prompt engineering for the Resume Analyzer.

Keeping prompts in one place (rather than inline in app.py) makes it easy to
version, test, and tune them independently of the UI/plumbing code - a good
practice to point to in an interview.
"""

SYSTEM_PROMPT = """You are the analysis engine behind TrueFit, an honest
resume-to-job fit tool. You act as an expert technical recruiter and career
coach with 15+ years of experience hiring for tech roles. You give honest,
specific, actionable feedback - not generic platitudes. You never invent
facts about the candidate that are not supported by their resume text.

A core part of your job is judging HOW LONG it would realistically take an
average motivated candidate to close a given skill gap - this is what makes
TrueFit different from a generic keyword matcher. Be realistic: a new
programming language syntax is quick to pick up; deep expertise in a
specialized domain (e.g. distributed systems, ML infra, compliance) is not.

You always respond with a single valid JSON object and nothing else -
no markdown code fences, no preamble, no explanation outside the JSON.
"""

# The main analysis prompt. We ask for a strict JSON schema so the Streamlit
# app can render structured UI (progress bars, tables, expandable sections)
# instead of just dumping a wall of text.
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following RESUME against the JOB DESCRIPTION.

<resume>
{resume_text}
</resume>

<job_description>
{jd_text}
</job_description>

Think step by step internally about:
1. What skills/technologies/experience the job actually requires (explicit and implied)
2. What skills/technologies/experience the resume demonstrates
3. Where they overlap and where there are gaps
4. How the resume's phrasing could be improved to better match this specific job

Then respond with ONLY a JSON object matching this exact schema:

{{
  "match_score": <integer 0-100, how well this resume fits this job>,
  "score_reasoning": "<2-3 sentences explaining the score>",
  "matched_skills": ["<skill the resume already demonstrates that the job wants>", ...],
  "skill_gaps": [
    {{
      "skill": "<a specific skill/requirement missing from the resume>",
      "priority": "<must_have|nice_to_have>",
      "learnability": "<quick|moderate|long>",
      "estimated_time": "<realistic human-readable estimate, e.g. '1-2 weeks', '2-3 months', '1+ years'>",
      "reason": "<one sentence on WHY it takes that long - e.g. 'syntax-level tool, learnable via docs' vs 'requires hands-on production experience'>"
    }}
  ],
  "strengths": ["<specific strength of this resume for this job>", ...],
  "weaknesses": ["<specific weakness or red flag for this job>", ...],
  "enhanced_bullets": [
    {{
      "original": "<a real bullet point copied from the resume>",
      "improved": "<rewritten version, ATS-friendly, quantified where possible, aligned to JD keywords - but do NOT invent numbers or facts not implied by the original>",
      "reason": "<one short sentence on why this is better for this JD>"
    }}
  ],
  "ats_keywords_missing": ["<exact keyword/phrase from the JD that is absent from the resume text and should probably be added if truthful>", ...],
  "learning_resources": [
    {{
      "skill": "<a skill name matching one from skill_gaps>",
      "resource": "<name of a specific, real, well-known resource (course, doc, book, certification)>",
      "type": "<course|documentation|book|certification|project idea>",
      "why": "<one sentence on why this resource fits>"
    }}
  ],
  "overall_summary": "<a candid 3-4 sentence summary of fit, tone: honest but constructive>"
}}

Rules:
- Provide 3-6 items for list fields where possible (fewer is fine if there genuinely isn't more to say).
- Provide 3-5 enhanced_bullets, using REAL bullets found in the resume text as "original".
- For skill_gaps: order does not matter, but be honest and specific about learnability - do not
  default everything to "moderate". Genuinely quick things (a new framework's syntax, a tool with
  good docs) should be marked "quick"; genuinely deep things (real production experience, a
  specialized domain, a certification with prerequisites) should be marked "long". This
  distinction is the main value of this tool, so take it seriously.
- Provide learning_resources for must_have skill_gaps first (prioritize those), then nice_to_have if room.
- Never fabricate resume content, employers, dates, or metrics that are not present or clearly implied.
- match_score should be a realistic, discriminating number - do not default to 70-85 for everything. Weak fits should score low.
- Return ONLY the JSON object. No markdown fences, no extra commentary.
"""


def build_analysis_prompt(resume_text: str, jd_text: str, additional_context: str = "") -> str:
    """Fill in the analysis prompt template with the actual resume and JD text.

    If additional_context is provided (e.g. answers from the recruiter
    follow-up interview), it's appended as a clearly-labeled supplementary
    block so the model treats it as trustworthy candidate-provided info,
    distinct from the original resume text.
    """
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        resume_text=resume_text.strip(),
        jd_text=jd_text.strip(),
    )
    if additional_context.strip():
        prompt += f"""

<candidate_followup_answers>
The candidate was also asked clarifying follow-up questions (like a real
recruiter would ask) and gave these answers. Treat this as truthful,
candidate-provided information - incorporate it into your analysis,
enhanced_bullets, and skill_gaps exactly as if it had been written on the
resume itself. If an answer reveals a skill or metric not on the original
resume, update matched_skills / remove or downgrade the relevant skill_gap
accordingly.

{additional_context.strip()}
</candidate_followup_answers>
"""
    return prompt


# ---------------------------------------------------------------------------
# Recruiter follow-up interview simulation
#
# Instead of a single-shot analysis, TrueFit looks at vague or unquantified
# claims and skill gaps, then asks the kind of targeted questions a real
# recruiter would ask in a screening call - the goal is to surface real
# achievements/experience the candidate didn't think to write down, before
# finalizing feedback.
# ---------------------------------------------------------------------------

FOLLOWUP_QUESTIONS_PROMPT_TEMPLATE = """You are simulating a recruiter's screening call.
You already analyzed this candidate's resume against a job description and
found some weaknesses and skill gaps. Before finalizing your feedback, you
want to ask the candidate a few sharp, specific follow-up questions - the
kind a good recruiter asks to find hidden experience that just wasn't
written down clearly, or to probe whether a vague claim is backed by real depth.

<resume>
{resume_text}
</resume>

<job_description>
{jd_text}
</job_description>

<prior_analysis_weaknesses>
{weaknesses}
</prior_analysis_weaknesses>

<prior_analysis_skill_gaps>
{skill_gaps}
</prior_analysis_skill_gaps>

Generate 3-4 follow-up questions. Good questions:
- Target a SPECIFIC vague bullet point or a SPECIFIC skill gap - not generic ("tell me about yourself" is bad).
- Try to surface unwritten experience (e.g. "You mention 'collaborated with backend teams' - did that involve designing or just consuming APIs?").
- Ask for missing quantification (team size, scale, timeframe, measurable outcome) where a bullet is vague.
- Probe whether a "missing" skill might actually exist but wasn't mentioned (e.g. "The role wants CI/CD experience - have you set up or maintained any pipelines, even informally?").

Respond with ONLY a JSON object matching this schema:

{{
  "questions": [
    {{
      "id": "q1",
      "question": "<the actual question to ask the candidate, conversational tone>",
      "targets": "<one short phrase: what this question is trying to uncover>"
    }}
  ]
}}

Return exactly 3-4 questions. No markdown fences, no extra commentary.
"""


def build_followup_questions_prompt(resume_text: str, jd_text: str, analysis: dict) -> str:
    """Build the prompt that generates recruiter-style follow-up questions,
    grounded in the specific weaknesses/gaps found in the initial analysis."""
    weaknesses = analysis.get("weaknesses", [])
    skill_gaps = analysis.get("skill_gaps", [])

    weaknesses_str = "\n".join(f"- {w}" for w in weaknesses) or "None listed."
    skill_gaps_str = "\n".join(
        f"- {g.get('skill', '')} ({g.get('priority', '')}, {g.get('learnability', '')})"
        for g in skill_gaps
    ) or "None listed."

    return FOLLOWUP_QUESTIONS_PROMPT_TEMPLATE.format(
        resume_text=resume_text.strip(),
        jd_text=jd_text.strip(),
        weaknesses=weaknesses_str,
        skill_gaps=skill_gaps_str,
    )


def format_qa_as_context(questions: list, answers: dict) -> str:
    """Format the recruiter follow-up Q&A pairs into a plain text block
    suitable for injecting into build_analysis_prompt's additional_context."""
    lines = []
    for q in questions:
        qid = q.get("id", "")
        answer = answers.get(qid, "").strip()
        if answer:
            lines.append(f"Q: {q.get('question', '')}\nA: {answer}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: Automated job search
#
# When scoring many live job postings against one resume, running the full
# analysis prompt per posting would be slow and wasteful. This lightweight
# "quick match" prompt gives just a score + one-line reasoning, fast enough
# to run across 10+ postings in a single search.
# ---------------------------------------------------------------------------

QUICK_MATCH_PROMPT_TEMPLATE = """You are quickly screening a job posting against a candidate's resume.

<resume>
{resume_text}
</resume>

<job_posting>
Title: {job_title}
Company: {company}
Description: {description}
</job_posting>

Respond with ONLY a JSON object matching this schema:

{{
  "match_score": <integer 0-100, realistic and discriminating - do not default to 60-80>,
  "one_line_reason": "<one short sentence on why this score, referencing something specific>"
}}

Return ONLY the JSON object. No markdown fences, no extra commentary.
"""


def build_quick_match_prompt(resume_text: str, job_title: str, company: str, description: str) -> str:
    """Build a fast, cheap prompt to score one job posting against a resume.
    Used by the automated job search feature to rank many postings quickly."""
    # Truncate long descriptions to keep this call fast and cheap.
    trimmed_description = (description or "").strip()[:2000]
    return QUICK_MATCH_PROMPT_TEMPLATE.format(
        resume_text=resume_text.strip(),
        job_title=job_title or "Unknown title",
        company=company or "Unknown company",
        description=trimmed_description or "No description provided.",
    )
