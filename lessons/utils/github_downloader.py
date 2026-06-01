import os
import time
from urllib.parse import urlparse

import requests


def _parse_github_folder_url(url: str):
    """
    Parses URLs of the form:
      https://github.com/{owner}/{repo}/tree/{branch}/{path...}
    and returns (owner, repo, branch, path_in_repo).
    """
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        raise ValueError("Only github.com URLs are supported.")
    parts = [p for p in parsed.path.split("/") if p]
    # Expecting: owner / repo / tree / branch / path...
    if len(parts) < 4 or parts[2] != "tree":
        raise ValueError("URL must be a GitHub 'tree' URL to a folder, e.g. https://github.com/owner/repo/tree/branch/path")
    owner = parts[0]
    repo = parts[1]
    branch = parts[3]
    path_in_repo = "/".join(parts[4:])  # may be empty if root
    return owner, repo, branch, path_in_repo


def _gh_api_list_dir(owner: str, repo: str, path_in_repo: str, ref: str, token: str | None = None):
    """
    Uses the GitHub Contents API to list items within a directory.
    Returns JSON list with entries (type: 'file'|'dir'), including 'name', 'path', 'download_url'.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path_in_repo}"
    params = {"ref": ref} if ref else {}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise RuntimeError("GitHub API rate limit hit. Set a GITHUB_TOKEN env var to increase limits.")
    r.raise_for_status()
    data = r.json()
    # The API returns an object (not a list) if the path is a file; we need a directory.
    if isinstance(data, dict) and data.get("type") != "dir":
        raise ValueError("The provided URL points to a file, not a folder.")
    return data  # list


def _download_file(url: str, dest_path: str, token: str | None = None, retries: int = 3):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))  # simple backoff


_BASE_REPO_PREFIX = "https://github.com/louisfb01/agent-course-notebooks/tree/main/"


def download_github_folder(relative_path: str, local_folder: str, recursive: bool = True):
    """
    Download all files from a *relative* folder path within the fixed public repo
    into `local_folder`.

    Examples of `relative_path`:
      - "notebooks/lesson_11/images"
      - "/notebooks/lesson_11/images"   (leading slash ok)

    Notes:
    - Any '..' path traversal is blocked.
    - If you accidentally pass a full GitHub URL, it must start with the fixed base.
    """
    # Normalize input: strip leading slashes and forbid traversal
    rel = relative_path.lstrip("/").strip()
    if not rel or rel == ".":
        raise ValueError("Provide a non-empty relative path to a folder inside the repo.")
    if any(part in ("..",) for part in rel.split("/")):
        raise ValueError("Path traversal ('..') is not allowed in relative_path.")

    # If user accidentally passes an absolute URL, allow only if within the same base
    if rel.startswith("https://github.com/"):
        if not rel.startswith(_BASE_REPO_PREFIX):
            raise ValueError("Absolute URLs are not allowed unless they belong to the fixed repository base.")
        folder_url = rel  # already full and within base
    else:
        folder_url = _BASE_REPO_PREFIX + rel

    owner, repo, branch, path_in_repo = _parse_github_folder_url(folder_url)
    token = os.environ.get("GITHUB_TOKEN")
    items = _gh_api_list_dir(owner, repo, path_in_repo, branch, token=token)

    for item in items:
        t = item.get("type")
        name = item.get("name")
        item_path = item.get("path")
        if t == "file":
            download_url = item.get("download_url") or f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item_path}"
            # Preserve structure inside local_folder, relative to starting folder
            dest_rel = os.path.relpath(item_path, start=path_in_repo) if path_in_repo else name
            dest_path = os.path.join(local_folder, dest_rel)
            _download_file(download_url, dest_path, token=token)
        elif t == "dir" and recursive:
            os.path.relpath(item_path, start=f"{owner}/{repo}") if False else item_path  # noop, just clarity
            # Call recursively using the relative path (keeps the "hidden" base)
            sub_relative_from_repo_root = item_path  # already relative to repo root
            # Strip the leading repo root if present (it's not), ensure it's relative to repo root:
            download_github_folder(sub_relative_from_repo_root, os.path.join(local_folder, name), recursive=True)
        else:
            continue
