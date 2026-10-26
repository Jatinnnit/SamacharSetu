# Verification record

Verified locally on 25 October 2026 using Windows, Python 3.11.9 and Node.js 24.13.1.

| Check | Result |
|---|---|
| TypeScript `npm run typecheck` | Passed |
| Production `npm run build` | Passed, Next.js 16.4.3 |
| Python `pytest` | 31 passed in 15.74 seconds |
| Frontend `node --test` | 4 passed |
| Browser reader/search/bookmark/comparison flow | Passed |
| Browser real NLP/embedding/n-gram/LSA/evaluation/mapping flow | Passed |
| Browser mobile navigation and overflow flow | Passed at 390 × 844 |
| Browser provider-outage state | Passed; no fixtures shown as live news |
| Actual GDELT live request | Returned 46 deduplicated English records for an India search |
| Desktop visual inspection | Homepage and populated NLP Lab checked at 1440 px width |
| Mobile visual inspection | Homepage and populated evaluation page checked at 390 px width |
| Model audit | Confusion-matrix totals and computed accuracy match saved reports; classifier SHA-256 values verified |
| Split audit | No normalized exact headline overlap among retained train/validation/test sets |

All four browser cases passed on the final code. The Lab case was rerun independently after correcting an ambiguous test locator; the other three already passed. A real persistence defect found during the first browser run was fixed: the initial URL-selected offline mode is now stored with the filters, so it survives navigation and reloads. Cold-start model loading is allowed a 15-second assertion window in browser tests.

One upstream warning remains: Starlette's TestClient deprecates its HTTPX adapter in favour of HTTPX2. The tests pass with the pinned installed versions. This warning does not occur in the browser interface and does not change inference results.

The first test attempt through the restricted shell could not locate the browser installation. Browser tests were completed with the installed Playwright Chromium runtime. No user browser state or personal profile was accessed.

The delivered archive excludes raw/prepared datasets. `test_split_integrity` will skip in a freshly extracted copy until `scripts/prepare_data.py --download` recreates them; it was run and passed during this verification. All remaining backend tests use the included models/resources, mocked provider responses or authored fixtures.

Live GDELT availability is time-dependent. The successful request confirms the adapter worked during this run; it is not a guarantee of future provider uptime. No manual publication, cloud deployment, external sharing or email was performed.
