#!/usr/bin/env python3
"""Regenerate the staged Nova notebook fixtures and their distributable archive."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
LESSONS_DIR = REPO_ROOT / "lessons"
NOVA_SERVER_DIR = LESSONS_DIR / "research_agent_part_2" / "mcp_server"
DEFAULT_SOURCE = LESSONS_DIR / "research_agent_part_2" / "data" / "sample_research_folder"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "nova_samples"
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "nova_samples.zip"
DEFAULT_WORKSPACE = REPO_ROOT / "nova_workspace" / "fixture_generation"

LESSON_STAGES = (
    "16_fastmcp",
    "17_data_ingestion",
    "18_research_loop",
    "19_final_outputs",
    "25_integrate_agents",
)

EMPTY_NOVA_DIRECTORIES = (
    "local_files_from_research",
    "urls_from_guidelines",
    "urls_from_guidelines_code",
    "urls_from_guidelines_youtube_videos",
    "urls_from_research",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Clean research folder used as fixture input.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory that will contain staged fixtures.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Zip archive written after fixture generation.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE, help="Disposable working directory used by the real pipeline.")
    parser.add_argument("--queries", type=int, default=3, help="Number of Perplexity queries to run for the Lesson 19 fixture.")
    return parser.parse_args()


def require_credentials() -> None:
    required = ["FIRECRAWL_API_KEY"]
    if (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() in ("true", "1", "yes"):
        required.append("GOOGLE_CLOUD_PROJECT")
    else:
        required.append("GOOGLE_API_KEY")
    if (os.environ.get("WEB_SEARCH_PROVIDER") or "perplexity").strip().lower() == "tavily":
        required.append("TAVILY_API_KEY")
    else:
        required.append("PPLX_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required credential(s): {names}. Add them to {NOVA_SERVER_DIR / '.env'} or the environment.")


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_clean_source(source: Path, destination: Path) -> None:
    if not (source / "article_guideline.md").is_file():
        raise FileNotFoundError(f"Missing article_guideline.md in fixture source: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".nova", "research.md"))


def snapshot(workspace: Path, output: Path, lesson: str) -> None:
    destination = output / lesson
    shutil.copytree(workspace, destination)


def ensure_nova_directories(workspace: Path) -> None:
    nova = workspace / ".nova"
    for relative_path in EMPTY_NOVA_DIRECTORIES:
        (nova / relative_path).mkdir(parents=True, exist_ok=True)


def validate_tool_result(name: str, result: dict[str, Any]) -> None:
    if result.get("status") not in {"success", "warning"}:
        raise RuntimeError(f"{name} failed: {result}")


def create_archive(samples_directory: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        root_arcname = Path(samples_directory.name)
        archive.writestr(f"{root_arcname.as_posix()}/", "")
        for path in sorted(samples_directory.rglob("*")):
            arcname = root_arcname / path.relative_to(samples_directory)
            if path.is_dir():
                archive.writestr(f"{arcname.as_posix()}/", "")
            else:
                archive.write(path, arcname.as_posix())


def verify_archive(archive_path: Path) -> None:
    required = {
        "nova_samples/16_fastmcp/article_guideline.md",
        "nova_samples/17_data_ingestion/article_guideline.md",
        "nova_samples/18_research_loop/.nova/guidelines_filenames.json",
        "nova_samples/19_final_outputs/.nova/perplexity_results.md",
        "nova_samples/25_integrate_agents/article_guideline.md",
    }
    required_directories = {f"nova_samples/18_research_loop/.nova/{name}/" for name in EMPTY_NOVA_DIRECTORIES}

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted((required | required_directories) - names)
        if missing:
            raise RuntimeError(f"Archive is missing required entries: {missing}")

        with tempfile.TemporaryDirectory(prefix="nova-samples-") as temporary_directory:
            archive.extractall(temporary_directory)
            extracted = Path(temporary_directory) / "nova_samples"
            if not (extracted / "18_research_loop" / ".nova").is_dir():
                raise RuntimeError("Archive round-trip lost the .nova directory.")
            for directory in required_directories:
                relative = Path(directory.removeprefix("nova_samples/").rstrip("/"))
                if not (extracted / relative).is_dir():
                    raise RuntimeError(f"Archive round-trip lost empty directory: {relative}")


async def generate(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    archive = args.archive.resolve()
    workspace = args.workspace.resolve()

    load_dotenv(NOVA_SERVER_DIR / ".env", override=False)
    require_credentials()

    sys.path.insert(0, str(LESSONS_DIR))
    from research_agent_part_2.mcp_server.src.tools import (  # noqa: PLC0415
        extract_guidelines_urls_tool,
        generate_next_queries_tool,
        process_github_urls_tool,
        process_local_files_tool,
        run_perplexity_research_tool,
        scrape_and_clean_other_urls_tool,
        transcribe_youtube_videos_tool,
    )

    reset_directory(output)
    reset_directory(workspace)
    copy_clean_source(source, workspace)

    for lesson in ("16_fastmcp", "17_data_ingestion", "25_integrate_agents"):
        snapshot(workspace, output, lesson)

    validate_tool_result("extract_guidelines_urls", extract_guidelines_urls_tool(str(workspace)))
    validate_tool_result("process_local_files", process_local_files_tool(str(workspace)))

    ingestion_results = await asyncio.gather(
        scrape_and_clean_other_urls_tool(str(workspace)),
        process_github_urls_tool(str(workspace)),
        transcribe_youtube_videos_tool(str(workspace)),
    )
    for name, result in zip(("scrape URLs", "process GitHub URLs", "transcribe YouTube URLs"), ingestion_results):
        validate_tool_result(name, result)

    ensure_nova_directories(workspace)
    snapshot(workspace, output, "18_research_loop")

    query_result = await generate_next_queries_tool(str(workspace), n_queries=args.queries)
    validate_tool_result("generate_next_queries", query_result)
    queries = [query for query, _reason in query_result["queries"]]
    validate_tool_result("run_perplexity_research", await run_perplexity_research_tool(str(workspace), queries))
    snapshot(workspace, output, "19_final_outputs")

    create_archive(output, archive)
    verify_archive(archive)

    print(f"Created staged fixtures: {output}")
    print(f"Created verified archive: {archive}")


def main() -> None:
    asyncio.run(generate(parse_args()))


if __name__ == "__main__":
    main()
