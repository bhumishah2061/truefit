# 🎯 TrueFit — Honest Resume-to-Job Fit Analysis

An LLM-powered tool that compares a resume against a specific job description
and returns a structured fit analysis: match score, skill gaps prioritized
by how long they'd realistically take to close, strengths & weaknesses,
ATS keyword gaps, rewritten bullet points, and a personalized learning plan.

Built with **Python, Streamlit, and a provider-agnostic LLM layer** (free Groq or Gemini by default; Anthropic Claude also supported).

## Why this project (and how it's different from every other "resume analyzer")

Most GenAI resume projects are a thin wrapper around one prompt: paste
resume, paste JD, get a score and a keyword list. That's the same demo
everyone builds. TrueFit is designed around one core insight that generic
tools miss: **not all skill gaps are equal.** Being missing "Kubernetes
production experience" and being missing "Tailwind CSS" are both "gaps,"
but one takes a weekend and the other takes months. A tool that treats them
the same isn't actually useful.

- **Recruiter follow-up interview simulation** — a static analysis can only
  work with what's already written down. After the first pass, TrueFit
  generates 3-4 targeted questions (like a real recruiter screening call
  would ask) aimed at specific vague bullets or skill gaps - then re-runs
  the full analysis incorporating your answers, so hidden experience you
  forgot to write down actually counts.
- **Learnability-weighted gap analysis** — every missing skill is tagged
  `quick` / `moderate` / `long` with a realistic time estimate and a reason,
  then the UI groups gaps into "quick wins" vs. "long-term investments" so
  the user knows what to fix *this week* before applying vs. what to plan
  for over months. This turns a static list into an actual action plan.
- **Structured output prompting** — the LLM is instructed to return a strict
  JSON schema, which the UI renders into progress bars, grouped columns, and
  expandable sections (not just a wall of text).
- **Constrained generation** — the prompt explicitly forbids fabricating
  resume facts (dates, employers, metrics) that aren't in the original text,
  which is a real, non-trivial LLM reliability problem.
- **Prompt/logic separation** — prompts live in `prompts.py`, independent of
  the UI (`app.py`) and file-parsing logic (`utils.py`), so each piece can be
  tested and improved independently.
- **Multi-format ingestion** — resumes can come in as PDF, DOCX, or plain
  text, normalized into a single text pipeline before hitting the model.

## Features

- 📄 Upload resume as PDF / DOCX / TXT (or paste text directly)
- 💼 Paste any job description
- 📊 Numeric match score with reasoning
- ✅ Strengths & ⚠️ weaknesses specific to that job
- ❌ Skill gaps grouped by **learnability**: ⚡ quick win / 🛠️ moderate effort / ⛰️ long-term investment, each with a time estimate and must-have/nice-to-have tag
- 🎤 **Recruiter follow-up interview** — simulated screening questions that surface hidden experience, then a re-run analysis with your answers factored in
- 🔑 Missing ATS keywords worth adding (if truthful)
- 🔎 **Automated job search** — search live job postings (via Adzuna) and TrueFit scores each one against your resume, ranking best-fit opportunities first
- ✍️ Rewritten resume bullet points aligned to the job description
- 📚 Curated learning resources for each skill gap
- ⬇️ Export full analysis as JSON

## Setup

1. **Clone/download this folder and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free API key** (no credit card) from one of:
   - [console.groq.com/keys](https://console.groq.com/keys) — recommended, very fast
   - [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — Google's free tier
   *(Optional: to use Claude instead, get a paid key from
   [console.anthropic.com](https://console.anthropic.com) and pick
   "Claude (paid)" in the sidebar.)*

3. **Set your API key** (either works):

   - Copy `.env.example` to `.env` and fill in your key, **or**
   - Enter it directly in the app's sidebar at runtime (session-only, never stored).

4. **(Optional, for Job Search tab) Get a free Adzuna API key** at
   [developer.adzuna.com](https://developer.adzuna.com) — no credit card
   needed. Enter the App ID and App Key in the sidebar under "Job Search
   (optional)" when you want to use that tab.

5. **Run the app:**

   ```bash
   streamlit run app.py
   ```

6. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   upload a resume, paste a job description, and click **Analyze Fit**.

## Project structure

```
truefit/
├── app.py           # Streamlit UI + orchestration
├── prompts.py        # Prompt engineering (system + analysis prompts + follow-up interview)
├── llm.py            # Provider abstraction (Groq / Gemini free, Claude optional)
├── job_search.py      # Live job search via the Adzuna API (Phase 2)
├── utils.py          # PDF/DOCX/TXT text extraction
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap / Phase 2 ideas

These are natural next steps if you want to extend the project further:

- **Cover letter generator**: draft a tailored cover letter per job posting.
- **Resume version history**: track how the match score improves as the
  resume is edited over multiple rounds.
- **Multi-model comparison**: run the same analysis through multiple models
  and show where they agree/disagree (great for a "prompt eval" angle).
- **Batch mode**: analyze one resume against many job descriptions at once
  and rank them by fit.

## Notes on responsible use

This tool is meant to help you understand and improve your resume — it
does not fabricate experience you don't have. Always review AI-suggested
bullet point rewrites for accuracy before using them.
