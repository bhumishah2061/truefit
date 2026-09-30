"""
TrueFit - Honest Resume-to-Job Fit Analysis

Phase 1:
- Resume upload / paste
- Job description analysis
- Match score
- Skill gaps
- Strengths / weaknesses
- Resume bullet improvements
- Learning resources
- Recruiter follow-up interview

Phase 2:
- Fetch live jobs from Himalayas
- Score jobs against the saved resume
- Rank jobs by match score
"""

import json
import os
import re

import streamlit as st
from dotenv import load_dotenv

from prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_followup_questions_prompt,
    build_quick_match_prompt,
    format_qa_as_context,
)

from job_search import search_jobs
from llm import PROVIDERS, LLMClient
from utils import extract_text_from_upload


load_dotenv()

st.set_page_config(
    page_title="TrueFit",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def get_client(api_key: str) -> LLMClient:
    provider = PROVIDERS[st.session_state["provider_label"]]
    return LLMClient(
        provider["id"],
        api_key,
        provider["default_model"],
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_llm_json(
    client: LLMClient,
    prompt: str,
    max_tokens: int = 4000,
) -> dict:

    raw_text = client.generate_json(
        SYSTEM_PROMPT,
        prompt,
        max_tokens,
    )

    cleaned = strip_code_fences(raw_text)

    return json.loads(cleaned)


def call_analysis(
    client: LLMClient,
    resume_text: str,
    jd_text: str,
    additional_context: str = "",
) -> dict:

    prompt = build_analysis_prompt(
        resume_text,
        jd_text,
        additional_context,
    )

    return call_llm_json(
        client,
        prompt,
        max_tokens=4000,
    )


def call_followup_questions(
    client: LLMClient,
    resume_text: str,
    jd_text: str,
    analysis: dict,
) -> dict:

    prompt = build_followup_questions_prompt(
        resume_text,
        jd_text,
        analysis,
    )

    return call_llm_json(
        client,
        prompt,
        max_tokens=1200,
    )


def call_quick_match(
    client: LLMClient,
    resume_text: str,
    job_title: str,
    company: str,
    description: str,
) -> dict:

    prompt = build_quick_match_prompt(
        resume_text,
        job_title,
        company,
        description,
    )

    return call_llm_json(
        client,
        prompt,
        max_tokens=300,
    )


def friendly_api_error(e: Exception) -> str:

    msg = str(e)
    low = msg.lower()

    if (
        "429" in msg
        or "resource_exhausted" in low
        or ("rate" in low and "limit" in low)
    ):
        return (
            "Rate limit hit. Wait about a minute and try again."
        )

    if (
        "api key" in low
        or "401" in msg
        or "403" in msg
        or "permission" in low
    ):
        return (
            "The API key looks invalid or does not have access. "
            "Please check the API key in the sidebar."
        )

    if "404" in msg or "not found" in low:
        return (
            "The selected model was not found. "
            f"Details: {msg}"
        )

    return f"Something went wrong: {msg}"


def score_color(score: int) -> str:

    if score >= 75:
        return "🟢"

    if score >= 50:
        return "🟡"

    return "🔴"


# ============================================================
# GAP SORTING
# ============================================================

LEARNABILITY_META = {
    "quick": {
        "emoji": "⚡",
        "label": "Quick win",
        "order": 0,
    },
    "moderate": {
        "emoji": "🛠️",
        "label": "Moderate effort",
        "order": 1,
    },
    "long": {
        "emoji": "⛰️",
        "label": "Long-term investment",
        "order": 2,
    },
}


def sort_gaps_by_learnability(gaps: list) -> list:

    def sort_key(gap):

        learn_order = LEARNABILITY_META.get(
            gap.get("learnability", "moderate"),
            LEARNABILITY_META["moderate"],
        )["order"]

        priority_order = (
            0
            if gap.get("priority") == "must_have"
            else 1
        )

        return learn_order, priority_order

    return sorted(gaps, key=sort_key)


# ============================================================
# ANALYSIS DISPLAY
# ============================================================

def render_analysis(result: dict, key_prefix: str):

    score = result.get("match_score", 0)

    top_l, top_r = st.columns([1, 3])

    with top_l:
        st.metric(
            "Match Score",
            f"{score_color(score)} {score}/100",
        )

    with top_r:
        st.write("**Why this score:**")
        st.write(
            result.get(
                "score_reasoning",
                "",
            )
        )

    st.progress(
        min(max(score, 0), 100) / 100
    )

    st.write("**Overall summary:**")

    st.info(
        result.get(
            "overall_summary",
            "",
        )
    )

    st.divider()

    sc1, sc2 = st.columns(2)

    with sc1:

        st.subheader("✅ Strengths")

        for strength in result.get("strengths", []):
            st.markdown(f"- {strength}")

        st.subheader("🧩 Skills You Already Match")

        matched = result.get(
            "matched_skills",
            [],
        )

        if matched:
            st.write(
                " · ".join(
                    f"`{skill}`"
                    for skill in matched
                )
            )
        else:
            st.caption("None detected.")

    with sc2:

        st.subheader("⚠️ Weaknesses")

        for weakness in result.get(
            "weaknesses",
            [],
        ):
            st.markdown(
                f"- {weakness}"
            )

    st.divider()

    st.subheader(
        "❌ Skill Gaps — Prioritized by How Long They Take to Close"
    )

    st.caption(
        "TrueFit shows not only what you're missing, "
        "but how realistic each gap is to close."
    )

    gaps = sort_gaps_by_learnability(
        result.get("skill_gaps", [])
    )

    if not gaps:

        st.caption(
            "No significant skill gaps detected."
        )

    else:

        gap_cols = st.columns(3)

        buckets = {
            "quick": [],
            "moderate": [],
            "long": [],
        }

        for gap in gaps:

            buckets.setdefault(
                gap.get(
                    "learnability",
                    "moderate",
                ),
                [],
            ).append(gap)

        for col, key in zip(
            gap_cols,
            ["quick", "moderate", "long"],
        ):

            meta = LEARNABILITY_META[key]

            with col:

                st.markdown(
                    f"**{meta['emoji']} {meta['label']}**"
                )

                items = buckets.get(
                    key,
                    [],
                )

                if not items:
                    st.caption("None")

                for gap in items:

                    priority_tag = (
                        "🔴 must-have"
                        if gap.get("priority")
                        == "must_have"
                        else "🟡 nice-to-have"
                    )

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{gap.get('skill', '')}**"
                        )

                        st.caption(
                            f"{priority_tag} · "
                            f"~{gap.get('estimated_time', 'unknown')}"
                        )

                        st.write(
                            gap.get(
                                "reason",
                                "",
                            )
                        )

    ats = result.get(
        "ats_keywords_missing",
        [],
    )

    if ats:

        st.subheader(
            "🔑 ATS Keywords Missing From Your Resume"
        )

        st.write(
            " · ".join(
                f"`{keyword}`"
                for keyword in ats
            )
        )

    st.divider()

    st.subheader(
        "✍️ Suggested Bullet Point Rewrites"
    )

    bullets = result.get(
        "enhanced_bullets",
        [],
    )

    for i, bullet in enumerate(
        bullets,
        1,
    ):

        with st.expander(
            f"Rewrite {i}: "
            f"{bullet.get('original', '')[:70]}..."
        ):

            st.markdown("**Original:**")

            st.write(
                bullet.get(
                    "original",
                    "",
                )
            )

            st.markdown("**Improved:**")

            st.success(
                bullet.get(
                    "improved",
                    "",
                )
            )

            st.caption(
                f"Why: {bullet.get('reason', '')}"
            )

    st.divider()

    st.subheader(
        "📚 Learning Resources for Missing Skills"
    )

    resources = result.get(
        "learning_resources",
        [],
    )

    for resource in resources:

        st.markdown(
            f"**{resource.get('skill', '')}** — "
            f"{resource.get('resource', '')} "
            f"*({resource.get('type', '')})*"
        )

        st.caption(
            resource.get(
                "why",
                "",
            )
        )

    st.divider()

    st.download_button(
        "⬇️ Download Full Analysis (JSON)",
        data=json.dumps(
            result,
            indent=2,
        ),
        file_name=(
            f"truefit_{key_prefix}_analysis.json"
        ),
        mime="application/json",
        key=f"download_{key_prefix}",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    provider_label = st.selectbox(
        "AI provider",
        list(PROVIDERS.keys()),
    )

    st.session_state[
        "provider_label"
    ] = provider_label

    provider = PROVIDERS[
        provider_label
    ]

    env_key = os.environ.get(
        provider["env_key"],
        "",
    )

    api_key = st.text_input(
        "API Key",
        value=env_key,
        type="password",
        help=provider["key_help"],
    )

    st.caption(
        provider["key_help"]
    )

    st.divider()

    st.markdown(
        "**About TrueFit**\n\n"
        "TrueFit analyzes your resume against "
        "specific jobs and shows strengths, "
        "skill gaps, ATS keywords and learning "
        "resources."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🎯 TrueFit")

st.caption(
    "Honest resume-to-job fit analysis — "
    "not just a score."
)

st.write(
    "Upload your resume and paste a job description "
    "to get a detailed fit analysis."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📄 Resume Analyzer",
        "🔎 Job Search",
    ]
)


# ============================================================
# TAB 1 — RESUME ANALYZER
# ============================================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📄 Your Resume"
        )

        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=[
                "pdf",
                "docx",
                "txt",
            ],
        )

        resume_text_manual = st.text_area(
            "...or paste resume text directly",
            height=200,
        )

    with col2:

        st.subheader(
            "💼 Job Description"
        )

        jd_text = st.text_area(
            "Paste the job description here",
            height=280,
        )

    analyze_clicked = st.button(
        "🔍 Analyze Fit",
        type="primary",
        use_container_width=True,
    )

    if analyze_clicked:

        if not api_key:

            st.error(
                "Please enter your API key in the sidebar."
            )

            st.stop()

        resume_text = ""

        if uploaded_file is not None:

            try:

                resume_text = (
                    extract_text_from_upload(
                        uploaded_file
                    )
                )

            except Exception as e:

                st.error(
                    f"Could not read uploaded resume: {e}"
                )

                st.stop()

        elif resume_text_manual.strip():

            resume_text = (
                resume_text_manual.strip()
            )

        if not resume_text:

            st.error(
                "Please upload a resume or paste resume text."
            )

            st.stop()

        if not jd_text.strip():

            st.error(
                "Please paste a job description."
            )

            st.stop()

        with st.spinner(
            "Analyzing resume..."
        ):

            try:

                client = get_client(
                    api_key
                )

                result = call_analysis(
                    client,
                    resume_text,
                    jd_text,
                )

            except json.JSONDecodeError:

                st.error(
                    "The model returned invalid JSON. "
                    "Please try again."
                )

                st.stop()

            except Exception as e:

                st.error(
                    friendly_api_error(e)
                )

                st.stop()

        st.session_state[
            "result"
        ] = result

        st.session_state[
            "resume_text"
        ] = resume_text

        st.session_state[
            "jd_text"
        ] = jd_text

        st.session_state.pop(
            "followup_questions",
            None,
        )

        st.session_state.pop(
            "refined_result",
            None,
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "result" in st.session_state:

        result = st.session_state[
            "result"
        ]

        st.divider()

        st.header(
            "📊 Results"
        )

        render_analysis(
            result,
            "initial",
        )

        # ----------------------------------------------------
        # FOLLOW-UP INTERVIEW
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🎤 Recruiter Follow-Up Interview"
        )

        st.write(
            "TrueFit can ask targeted recruiter-style "
            "questions and refine the analysis."
        )

        generate_questions_clicked = st.button(
            "🎙️ Start Follow-Up Interview"
        )

        if generate_questions_clicked:

            if not api_key:

                st.error(
                    "Please enter your API key."
                )

                st.stop()

            with st.spinner(
                "Thinking of recruiter questions..."
            ):

                try:

                    client = get_client(
                        api_key
                    )

                    followup = (
                        call_followup_questions(
                            client,
                            st.session_state[
                                "resume_text"
                            ],
                            st.session_state[
                                "jd_text"
                            ],
                            result,
                        )
                    )

                    st.session_state[
                        "followup_questions"
                    ] = followup.get(
                        "questions",
                        [],
                    )

                except Exception as e:

                    st.error(
                        friendly_api_error(e)
                    )

                    st.stop()

        if (
            "followup_questions"
            in st.session_state
        ):

            questions = st.session_state[
                "followup_questions"
            ]

            with st.form(
                "followup_form"
            ):

                for question in questions:

                    qid = question.get(
                        "id",
                        "",
                    )

                    st.markdown(
                        f"**{question.get('question', '')}**"
                    )

                    st.text_area(
                        "Your answer",
                        key=f"answer_{qid}",
                        height=80,
                        label_visibility="collapsed",
                    )

                refine_clicked = (
                    st.form_submit_button(
                        "🔁 Refine Analysis",
                        type="primary",
                    )
                )

            if refine_clicked:

                answers = {
                    q.get("id", ""):
                    st.session_state.get(
                        f"answer_{q.get('id', '')}",
                        "",
                    )
                    for q in questions
                }

                qa_context = (
                    format_qa_as_context(
                        questions,
                        answers,
                    )
                )

                if not qa_context.strip():

                    st.warning(
                        "Please answer at least one question."
                    )

                else:

                    with st.spinner(
                        "Re-analyzing..."
                    ):

                        try:

                            client = get_client(
                                api_key
                            )

                            refined_result = (
                                call_analysis(
                                    client,
                                    st.session_state[
                                        "resume_text"
                                    ],
                                    st.session_state[
                                        "jd_text"
                                    ],
                                    additional_context=qa_context,
                                )
                            )

                            st.session_state[
                                "refined_result"
                            ] = refined_result

                        except Exception as e:

                            st.error(
                                friendly_api_error(e)
                            )

        if (
            "refined_result"
            in st.session_state
        ):

            st.divider()

            st.header(
                "🔄 Updated Results"
            )

            render_analysis(
                st.session_state[
                    "refined_result"
                ],
                "refined",
            )


# ============================================================
# TAB 2 — LIVE JOB SEARCH
# ============================================================

with tab2:

    st.header(
        "🔎 Automated Job Search"
    )

    st.write(
        "Search live job postings and compare them "
        "against the resume you analyzed."
    )

    if "resume_text" not in st.session_state:

        st.warning(
            "First upload/analyze your resume in "
            "the 📄 Resume Analyzer tab."
        )

    else:

        col1, col2, col3 = st.columns(
            [2, 2, 1]
        )

        with col1:

            job_keywords = st.text_input(
                "Job title / keywords",
                placeholder=(
                    "e.g. Python, Data Analyst, Backend"
                ),
            )

        with col2:

            job_location = st.text_input(
                "Country",
                value="India",
            )

        with col3:

            job_limit = st.selectbox(
                "Number of jobs",
                [5, 10, 20],
                index=0,
            )

        search_clicked = st.button(
            "🔍 Search Live Jobs",
            type="primary",
            use_container_width=True,
        )

        if search_clicked:

            if not job_keywords.strip():

                st.error(
                    "Please enter a job title or keyword."
                )

                st.stop()

            with st.spinner(
                "Fetching live jobs..."
            ):

                try:

                    jobs = search_jobs(
                        query=job_keywords.strip(),
                        country=job_location.strip(),
                        limit=job_limit,
                    )

                except Exception as e:

                    st.error(
                        f"Could not fetch jobs: {e}"
                    )

                    st.stop()

            if not jobs:

                st.info(
                    "No jobs found. Try broader keywords."
                )

            else:

                st.success(
                    f"Found {len(jobs)} live jobs."
                )

                # Save jobs for later AI scoring
                st.session_state[
                    "live_jobs"
                ] = jobs


    # --------------------------------------------------------
    # DISPLAY LIVE JOBS
    # --------------------------------------------------------

    if "live_jobs" in st.session_state:

        st.divider()

        st.subheader(
            "📋 Live Job Results"
        )

        for job in st.session_state[
            "live_jobs"
        ]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {job.get('title', 'Untitled role')}"
                )

                st.write(
                    f"**Company:** "
                    f"{job.get('company', 'Unknown')}"
                )

                location = job.get(
                    "location",
                    "",
                )

                if location:

                    st.write(
                        f"📍 **Location:** {location}"
                    )

                employment = job.get(
                    "employment_type",
                    "",
                )

                if employment:

                    st.write(
                        f"💼 **Type:** {employment}"
                    )

                description = job.get(
                    "description",
                    "",
                )

                if description:

                    clean_description = re.sub(
                        r"<[^>]+>",
                        " ",
                        description,
                    )

                    clean_description = re.sub(
                        r"\s+",
                        " ",
                        clean_description,
                    ).strip()

                    st.write(
                        clean_description[:500]
                        + (
                            "..."
                            if len(clean_description) > 500
                            else ""
                        )
                    )

                application_link = job.get(
                    "application_link",
                    "",
                )

                if application_link:

                    st.link_button(
                        "View & Apply →",
                        application_link,
                    )

        # ----------------------------------------------------
        # AI SCORING
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 AI Job Matching"
        )

        st.write(
            "Once your AI provider/API key is available, "
            "TrueFit can score these jobs against your resume."
        )

        score_jobs_clicked = st.button(
            "🎯 Score Jobs Against My Resume",
            type="primary",
            use_container_width=True,
        )

        if score_jobs_clicked:

            if not api_key:

                st.error(
                    "Please enter your AI provider API key "
                    "in the sidebar."
                )

                st.stop()

            client = get_client(
                api_key
            )

            scored_jobs = []

            progress = st.progress(
                0,
                text="Scoring jobs...",
            )

            jobs = st.session_state[
                "live_jobs"
            ]

            for i, job in enumerate(
                jobs
            ):

                try:

                    description = job.get(
                        "description",
                        "",
                    )

                    match = call_quick_match(
                        client,
                        st.session_state[
                            "resume_text"
                        ],
                        job.get(
                            "title",
                            "",
                        ),
                        job.get(
                            "company",
                            "",
                        ),
                        description,
                    )

                    job["match_score"] = int(
                        match.get(
                            "match_score",
                            0,
                        )
                    )

                    job["one_line_reason"] = (
                        match.get(
                            "one_line_reason",
                            "",
                        )
                    )

                except Exception as e:

                    job["match_score"] = None

                    job["one_line_reason"] = (
                        "Could not score this job."
                    )

                scored_jobs.append(
                    job
                )

                progress.progress(
                    (i + 1) / len(jobs),
                    text=(
                        f"Scored "
                        f"{i + 1}/{len(jobs)} jobs..."
                    ),
                )

            progress.empty()

            scored_jobs.sort(
                key=lambda job: (
                    job.get(
                        "match_score"
                    )
                    if job.get(
                        "match_score"
                    ) is not None
                    else -1
                ),
                reverse=True,
            )

            st.session_state[
                "job_search_results"
            ] = scored_jobs

    # --------------------------------------------------------
    # RANKED RESULTS
    # --------------------------------------------------------

    if "job_search_results" in st.session_state:

        st.divider()

        st.subheader(
            "🏆 Ranked Job Results"
        )

        for job in st.session_state[
            "job_search_results"
        ]:

            score = job.get(
                "match_score"
            )

            if score is not None:

                score_display = (
                    f"{score_color(score)} "
                    f"{score}/100"
                )

            else:

                score_display = "N/A"

            with st.container(
                border=True
            ):

                left, right = st.columns(
                    [4, 1]
                )

                with left:

                    st.markdown(
                        f"**{job.get('title', '')}**"
                    )

                    st.caption(
                        f"{job.get('company', '')} "
                        f"· "
                        f"{job.get('location', '')}"
                    )

                with right:

                    st.metric(
                        "Fit",
                        score_display,
                    )

                reason = job.get(
                    "one_line_reason",
                    "",
                )

                if reason:

                    st.write(
                        f"💡 {reason}"
                    )

                application_link = job.get(
                    "application_link",
                    "",
                )

                if application_link:

                    st.link_button(
                        "View & Apply →",
                        application_link,
                    )