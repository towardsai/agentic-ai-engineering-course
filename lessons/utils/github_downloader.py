import os
import time
import requests
from urllib.parse import urlparse

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
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))  # simple backoff

def download_github_folder(folder_url: str, local_folder: str, recursive: bool = True):
    """
    Download all files from a PUBLIC GitHub folder URL into `local_folder`.
    - folder_url: e.g. "https://github.com/owner/repo/tree/branch/path/to/folder"
    - local_folder: local destination folder (will be created if it doesn't exist)
    - recursive: if True, descends into subfolders

    Notes:
    - If you hit rate limits, set an environment variable GITHUB_TOKEN with a
      personal access token (no scopes required for public repos) to increase limits.
    """
    owner, repo, branch, path_in_repo = _parse_github_folder_url(folder_url)
    token = os.environ.get("GITHUB_TOKEN")

    items = _gh_api_list_dir(owner, repo, path_in_repo, branch, token=token)

    for item in items:
        item_type = item.get("type")
        item_name = item.get("name")
        item_path = item.get("path")  # path within repo
        if item_type == "file":
            download_url = item.get("download_url")
            if not download_url:
                # Fallback to raw URL
                download_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item_path}"
            dest_path = os.path.join(local_folder, os.path.relpath(item_path, start=path_in_repo) if path_in_repo else item_name)
            _download_file(download_url, dest_path, token=token)
        elif item_type == "dir" and recursive:
            # Recurse into subdirectory
            sub_folder_url = f"https://github.com/{owner}/{repo}/tree/{branch}/{item_path}"
            # Preserve subfolder structure inside local_folder
            sub_local_folder = os.path.join(local_folder, os.path.relpath(item_path, start=path_in_repo) if path_in_repo else item_name)
            download_github_folder(sub_folder_url, sub_local_folder, recursive=True)
        else:
            # Skip symlinks, submodules, etc.
            continue