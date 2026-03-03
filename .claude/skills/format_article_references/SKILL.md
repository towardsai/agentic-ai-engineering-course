---
name: format_article_references
description: Validate and format article references in APA 7th edition style. Checks URLs for 404s, discards broken links, re-numbers references, and ensures inline citations match.
user_invocable: true
---

You are an expert technical editor who specializes in reference management for articles. Your job is to validate, clean, and format the references section of a markdown article following APA 7th edition style, ensuring all links are live and all inline citations are consistent.

## Prerequisites

- `httpx` is installed via `uv add httpx` (should already be in `pyproject.toml`)
- Run the validation script with `uv run python`

## Chain of Thoughts

### Step 1 — Identify the article

If no article path is provided as an argument or through the currently opened file, ask the user for it. The input must be a markdown (`.md`) file.

### Step 2 — Read the article

Read the full article to understand:
- Where the **References section** is (look for `## References` or a heading with a similar name like `## Sources`, `## Bibliography`, `## Works Cited`).
- All **inline citations** in the article body (everything above the References section).

### Step 3 — Parse the References section

Extract every reference entry from the References section. Each entry follows this pattern:

```
N. Author Name. (Date). Full Title. Source. URL
```

For each entry, extract:
- **Number** (the sequential index)
- **Full text** (the complete APA citation line)
- **URL** (the link at the end of the entry)

Ignore any lines in the References section that are not numbered reference entries.

**IMPORTANT:** Only process links from the References section (or equivalent). Do NOT process links from other parts of the article (e.g., embedded images, regular hyperlinks in body text, CTA links, etc.).

### Step 4 — Parse and normalize inline citations

Scan the entire article body (everything above the References section) for inline citations. The canonical format is `[[N]](URL)`, but articles may contain malformed variants that must be detected and corrected.

**Detect all of these patterns** (where N is a citation number and URL is the reference link):

| Pattern | Example | Action |
|---------|---------|--------|
| `[[N]](URL)` | `[[1]](https://example.com)` | Already correct — keep as-is |
| `[N](URL)` | `[1](https://example.com)` | Missing outer brackets — fix to `[[N]](URL)` |
| `[[N]](URL)` but URL is wrong/mismatched | `[[1]](https://wrong-url.com)` | Fix the URL to match the reference |
| `[[N]]` (no URL) | `[[1]]` | Add the URL from the References section: `[[N]](URL)` |
| `[N]` (no URL, single brackets) | `[1]` — only when N clearly refers to a reference number | Add double brackets and URL: `[[N]](URL)` |

**Detection heuristic for bare `[N]`:** A bare `[N]` is a citation reference when:
- N is a number that matches an existing reference number in the References section
- It is NOT part of a markdown link `[text](url)` where text happens to be a number
- It is NOT inside a list item marker, code block, or image syntax

For each citation found (in any format), extract:
- **Number** (N)
- **URL** (if present)

Build a mapping of citation number to URL and normalize every citation to `[[N]](URL)`.

### Step 5 — Validate URLs and extract metadata

Run the validation script on all URLs from the References section. The script performs a **streaming GET** for each URL (reading only the first 32 KB — the `<head>` section) to simultaneously validate the link and extract HTML metadata.

```bash
uv run python .claude/skills/format_article_references/scripts/validate_urls.py <url1> <url2> ... <urlN>
```

The script outputs a JSON array. Each entry contains:
- `url`: the original URL
- `status_code`: HTTP status code (or null if unreachable)
- `valid`: boolean — true if status < 400
- `error`: error description if the request failed
- `metadata`: object with extracted fields:
  - `title`: from `<title>` or `og:title` meta tag (may be null)
  - `author`: from `meta[name=author]` or `article:author` (may be null)
  - `published_date`: from `article:published_time` or `meta[name=date]` (may be null)
  - `site_name`: from `og:site_name` (may be null)

### Step 6 — Report validation results

Present the results to the user in a clear table:

```
| # | URL | Status | Title | Author | Date | Site | Action |
|---|-----|--------|-------|--------|------|------|--------|
| 1 | https://example.com/article | 200 OK | My Article | J. Smith | 2025-01-15 | Example Blog | Keep |
| 2 | https://broken.com/page | 404 Not Found | — | — | — | — | Remove |
| 3 | https://timeout.com/slow | Timeout | — | — | — | — | Remove |
```

Ask the user to confirm before proceeding. The user may override decisions (e.g., keep a URL that timed out if they know it works).

### Step 7 — Remove broken references

For each reference marked for removal:
1. Remove the entry from the References section.
2. Remove all corresponding inline citations `[[N]](URL)` from the article body. When removing an inline citation, clean up any leftover formatting artifacts (double spaces, dangling commas, etc.).

### Step 8 — Re-number references

After removing broken entries, re-number the remaining references sequentially starting from 1. For example, if references 1, 3, and 5 survived, they become 1, 2, and 3.

Update BOTH:
- The reference numbers in the References section (e.g., `1.`, `2.`, `3.`)
- All inline citations `[[N]](URL)` in the article body to match the new numbering

### Step 9 — Verify existing citation information against metadata

For each surviving reference, compare what is **currently written** in the reference entry (author, date, title, source) against what was **extracted from the HTML metadata** in Step 5. This catches stale, incorrect, or fabricated citation details.

For each reference, parse the existing entry into its APA components and check each field:

| Field | Check | Example discrepancy |
|-------|-------|---------------------|
| **Author** | Does the existing author match `metadata.author`? | Entry says "Smith, J." but metadata says "Johnson, A." |
| **Date** | Does the existing date match `metadata.published_date`? | Entry says "(2024, March 10)" but metadata says "2025-01-28" |
| **Title** | Does the existing title match `metadata.title`? | Entry says "Evals Guide" but metadata says "Evals Are NOT All You Need" |
| **Source** | Does the existing source match `metadata.site_name`? | Entry says "Medium" but metadata says "O'Reilly Media" |

**Comparison rules:**
- Comparisons are **case-insensitive** and **whitespace-normalized**.
- For author names, match on last name at minimum (metadata may return "Hamel Husain" while the entry has "Husain, H." — these match).
- For titles, ignore minor trailing differences caused by site name suffixes (e.g., `"My Article - Blog Name"` vs `"My Article"`).
- For dates, compare only the date components that are available in metadata. If metadata has a full ISO timestamp, extract just the date. If the existing entry says `(n.d.)` but metadata provides a date, flag it.
- When metadata is null/missing for a field, **skip that field** — you cannot verify what you do not have. Do not flag these as discrepancies.

Present a discrepancy report to the user:

```
| # | Field | Current value | Metadata value | Action |
|---|-------|---------------|----------------|--------|
| 2 | Author | Smith, J. | Johnson, A. | Correct → Johnson, A. |
| 2 | Date | (n.d.) | 2025-01-28 | Correct → (2025, January 28) |
| 4 | Title | Evals Guide | Evals Are NOT All You Need | Correct → Evals Are NOT All You Need |
| 5 | Source | Medium | O'Reilly Media | Correct → O'Reilly Media |
```

If there are no discrepancies, report that all citation information is accurate and move on.

If there are discrepancies, ask the user to confirm corrections before applying them. The user may override any suggested correction (e.g., prefer a manually curated author name over what metadata returns).

### Step 10 — Format references in APA 7th edition using extracted metadata

Use the metadata returned by the script in Step 5 — incorporating any corrections confirmed in Step 9 — as the **primary source** for each APA field. Fall back to inference only when metadata is missing.

Ensure every reference entry in the References section follows APA 7th edition format:

```
N. Author Name. (Date). Full Title. Source. URL
```

For each field, apply this resolution order:

| Field | Primary source (from metadata) | Fallback |
|-------|-------------------------------|----------|
| **Author** | `metadata.author` | Use site/organization name inferred from URL domain |
| **Date** | `metadata.published_date` — format as `(YYYY, Month DD)` | `(n.d.)` |
| **Full Title** | `metadata.title` (prefer `og:title` — it is usually cleaner than `<title>`) | Infer from URL path: convert hyphens/underscores to spaces, capitalize |
| **Source** | `metadata.site_name` | Infer from domain (e.g., `philschmid.de` → "Philschmid", `youtube.com` → "YouTube") |
| **URL** | The raw URL (not wrapped in markdown link syntax) | — |

**Date formatting rules:**
- If `metadata.published_date` is an ISO 8601 string (e.g., `2025-01-15T10:30:00Z`), format as `(2025, January 15)`.
- If only year and month are available, format as `(2025, January)`.
- If only year is available, format as `(2025)`.
- If no date at all, use `(n.d.)`.

**Title cleanup:**
- Strip trailing site names that some `<title>` tags include (e.g., `"My Article | Example Blog"` → `"My Article"`).
- Strip leading/trailing whitespace and normalize internal whitespace.

For inline citations within the article body, keep the canonical format: `[[N]](URL)`.

### Step 11 — Cross-check inline citations vs references

Verify that:
1. Every inline citation `[[N]](URL)` in the body has a matching entry with the **same number and URL** in the References section.
2. Every reference in the References section is cited at least once in the body (warn about orphaned references but do not auto-remove them — ask the user).
3. Citation numbers are consecutive (1, 2, 3, ..., N) with no gaps.

If any mismatches are found, fix them and report what was corrected.

### Step 12 — Write the updated article

Apply all changes to the article file:
- Normalized inline citations (all in `[[N]](URL)` format)
- Updated References section with valid, re-numbered, APA-formatted entries (each separated by a blank line)
- Updated inline citations with corrected numbers
- Clean formatting (no double spaces, no dangling punctuation from removed citations)

### Step 13 — Report summary

Present a final summary:

```
References updated:
- Total original references: X
- Removed (broken URLs): Y
- Final references: Z
- Inline citations normalized: N (format corrections applied)
- Citation info corrected (from metadata): C fields across R references
- Inline citation numbers updated: W
- Metadata extracted from HTML: M out of Z references
```

## APA 7th Edition Reference Format

```
N. Author Last, First Initial. (YYYY, Month DD). Full Title of Article. Source Name. URL
```

### Examples

```
1. Iusztin, P. (2025, July 22). Context Engineering Guide 101. Decoding AI Magazine. https://decodingml.substack.com/p/context-engineering-2025s-1-skill

2. Muscalagiu, A. I. (2025, August 19). Scaling your AI enterprise architecture with MCP systems. Decoding AI Magazine. https://decodingml.substack.com/p/why-mcp-breaks-old-enterprise-ai

3. Use the Functional API. (n.d.). LangChain. https://langchain-ai.github.io/langgraph/how-tos/use-functional-api/

4. karpathy. (n.d.). X. https://x.com/karpathy/status/1937902205765607626
```

### Special Cases

| Scenario | Handling |
|----------|----------|
| Author unknown | Use the page/site name as author |
| Date unknown | Use `(n.d.)` |
| Title unknown | Infer from URL path, converting hyphens to spaces and capitalizing |
| Source unknown | Infer from domain name |
| URL with query parameters | Keep the full URL as-is |

## Inline Citation Format

Citations in the article body use double square brackets with a link:

```markdown
some text [[1]](https://example.com/source), [[2]](https://other.com/source).
```

Multiple citations can appear in sequence, separated by commas or spaces.

## Edge Cases

| Case | Handling |
|------|----------|
| No References section found | Inform the user and stop |
| All URLs are valid | Skip removal steps, still format in APA 7th edition |
| All URLs are broken | Warn the user, ask for confirmation before removing all |
| Inline citation with no matching reference | Flag to the user, ask how to handle |
| Reference with no inline citation | Warn user about orphaned reference, ask whether to keep or remove |
| Duplicate references (same URL, different numbers) | Merge into one, update all inline citations to use the surviving number |
| URL returns 403 Forbidden | Treat as valid (access may be restricted but page exists) |
| URL returns 5xx Server Error | Report as potentially temporary, ask user |
| URL times out | Report as broken, but let user override |

## Key Rules

- **ONLY** process URLs from the References section. Never add body links to references.
- Always ask user confirmation before removing references.
- Preserve the original article content and formatting outside of citation/reference changes.
- When removing an inline citation, ensure the surrounding text reads naturally (fix spacing, punctuation).
- References in the final output must be separated by blank lines for readability.
- References MUST use a **numbered list** format (`1. Author...`, `2. Author...`), NOT bullet points (`- [1] Author...`). The number is part of the markdown list syntax itself (e.g., `1.`), not a bracketed prefix.
- The URL in the References section is a raw URL (not markdown link syntax). Inline citations use `[[N]](URL)` format.
