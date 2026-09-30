from job_search import search_jobs

jobs = search_jobs(
    query="python",
    country="India",
    limit=5
)

print("Jobs found:", len(jobs))

for i, job in enumerate(jobs, start=1):
    print(f"{i}. {job['title']} | {job['company']} | {job['location']}")
    