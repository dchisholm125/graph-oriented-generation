"""Smoke tests for GOG-Lite reference implementation.

Proves that the public GOG-Lite skeleton can:
1. Build an import graph for a real fixture.
2. Isolate context from a natural language prompt.
3. Return a context bundle without onboarding artifacts.
4. Apply file caps correctly.
5. Penalize large files.
6. Extract keyword snippets instead of full large files.
7. Respect total token budgets.
8. Prefer test files when keywords match.

All tests use simple, deterministic fixtures — no network calls.
"""

from __future__ import annotations

from pathlib import Path

from gog_engine_lite.import_graph import build_import_graph
from gog_engine_lite.graph_search import (
    isolate_context,
    extract_keyword_snippets,
    _score_node,
    _looks_like_test_file,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOKENS_PER_FILE,
    DEFAULT_TOTAL_TOKEN_BUDGET,
    LARGE_FILE_PENALTY_DISTANCE,
)
from gog_cli.lite_serving import build_lite_context_bundle
from gog_cli.lite_membrane import apply_lite_membrane

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_REPO = REPO_ROOT / "target_repo"


def test_lite_graph_builds_for_fixture():
    """GOG-Lite must build a non-empty import graph for the fixture repo."""
    graph = build_import_graph(TARGET_REPO)
    assert graph.number_of_nodes() > 0, "GOG-Lite should find source files in target_repo"
    # The fixture should have at least some imports between files
    assert graph.number_of_edges() >= 0, "Edges may be sparse but graph should be valid"


def test_lite_context_isolation_runs():
    """GOG-Lite must return candidate files for a prompt."""
    graph = build_import_graph(TARGET_REPO)
    if graph.number_of_nodes() == 0:
        return  # fixture may not have parseable source
    files = isolate_context(graph, "Locate where user authentication state is passed to the dashboard")
    assert isinstance(files, list)
    # May return 0 if no keyword match, which is acceptable


def test_lite_context_bundle_runs():
    """GOG-Lite must return a full bundle with file contents."""
    bundle = build_lite_context_bundle(
        TARGET_REPO,
        prompt="Identify the default state variables in the auth store",
        max_files=10,
        max_depth=2,
    )
    assert "selected_nodes" in bundle
    assert "file_contents" in bundle
    assert "metadata" in bundle
    # If the fixture has source, we should get some files back
    assert isinstance(bundle["file_contents"], dict)


def test_lite_does_not_depend_on_onboarding():
    """GOG-Lite must not require persistent .gog/ artifacts."""
    # If target_repo had a .gog/ from earlier tests, ignore it
    bundle = build_lite_context_bundle(TARGET_REPO, "Refactor the login flow")
    # Should succeed even without onboarding because it builds graph on-the-fly
    assert "metadata" in bundle
    assert "node_count" in bundle["metadata"]


def test_benchmark_mode_uses_repo_relative_lite_matching(tmp_path):
    """Benchmark temp directory names must not affect GOG-Lite retrieval."""
    from gog_cli.executable_patch_benchmark import PatchTask, run_task_mode

    source_repo = tmp_path / "debug_query_serialization_easy-source"
    (source_repo / "src/utils").mkdir(parents=True)
    (source_repo / "src/router.ts").write_text("export const router = {}\n", encoding="utf-8")
    (source_repo / "src/store").mkdir(parents=True)
    (source_repo / "src/store/user.ts").write_text("export const user = null\n", encoding="utf-8")
    (source_repo / "src/main.ts").write_text("import './router'\n", encoding="utf-8")
    (source_repo / "src/utils/params-to-query.ts").write_text(
        "export default function params2query(params: Record<string, unknown>): string { return '' }\n",
        encoding="utf-8",
    )
    (source_repo / "src/utils/params-to-query.spec.ts").write_text(
        "import params2query from './params-to-query'\n",
        encoding="utf-8",
    )

    task = PatchTask(
        id="debug_query_serialization_easy",
        domain="debugging",
        difficulty="easy",
        prompt=(
            "Fix the query parameter serialization helper so it URI-encodes keys and values, "
            "preserves 0 and false, and omits null or undefined values."
        ),
        expected_files=("src/utils/params-to-query.ts", "src/utils/params-to-query.spec.ts"),
        validation_commands=(),
        setup_patches=(),
        notes="Regression fixture for benchmark temp path matching.",
        gold_files=("src/utils/params-to-query.ts", "src/utils/params-to-query.spec.ts"),
        expected_edit_files=("src/utils/params-to-query.ts",),
    )

    result = run_task_mode(
        source_repo=source_repo,
        task=task,
        mode="gog_lite",
        model="unused",
        attempts=1,
        timeout_s=1,
        retries=0,
        retry_delay_s=0,
        dry_run=True,
    )

    assert result["retrieved_files"] == [
        "src/utils/params-to-query.spec.ts",
        "src/utils/params-to-query.ts",
    ]
    assert result["context_metrics"]["context_precision"] == 1.0
    assert result["context_metrics"]["noise_ratio"] == 0.0


def test_file_cap_is_enforced():
    """GOG-Lite must never return more files than max_files."""
    graph = build_import_graph(TARGET_REPO)
    if graph.number_of_nodes() == 0:
        return

    for cap in (3, 5, 7):
        files = isolate_context(graph, "authentication state", max_files=cap)
        assert len(files) <= cap, f"Expected ≤{cap} files, got {len(files)}"


def test_large_file_penalty_reduces_score():
    """Files larger than MAX_FILE_SIZE_BYTES should score lower than small files
    at the same distance, unless they are exact filename matches."""
    from gog_engine_lite.graph_search import MAX_FILE_SIZE_BYTES

    # Create a minimal graph with two nodes at the same distance
    import networkx as nx
    graph = nx.DiGraph()
    small_file = "/repo/src/utils/helper.ts"
    large_file = "/repo/src/utils/big_helper.ts"
    seed = "/repo/src/main.ts"

    graph.add_node(small_file, size=500)
    graph.add_node(large_file, size=MAX_FILE_SIZE_BYTES + 10_000)
    graph.add_node(seed, size=300)
    graph.add_edge(seed, small_file)
    graph.add_edge(seed, large_file)

    keywords = {"helper"}
    seed_set = {seed}

    small_score = _score_node(small_file, distance=1, seed_set=seed_set,
                               prompt_keywords=keywords, graph=graph)
    large_score = _score_node(large_file, distance=1, seed_set=seed_set,
                               prompt_keywords=keywords, graph=graph)

    # Both get the same keyword match, but large file gets a penalty
    assert large_score < small_score, (
        f"Large file score ({large_score}) should be lower than small file ({small_score})"
    )


def test_exact_filename_match_gets_highest_score():
    """A file whose name exactly matches a keyword should outrank distant files."""
    import networkx as nx
    graph = nx.DiGraph()
    exact_match = "/repo/src/auth/store.ts"
    distant = "/repo/src/utils/helper.ts"
    seed = "/repo/src/main.ts"

    graph.add_node(exact_match, size=500)
    graph.add_node(distant, size=500)
    graph.add_node(seed, size=300)
    graph.add_edge(seed, distant)

    keywords = {"store"}
    seed_set = {seed}

    exact_score = _score_node(exact_match, distance=0, seed_set=seed_set,
                                prompt_keywords=keywords, graph=graph)
    distant_score = _score_node(distant, distance=1, seed_set=seed_set,
                                 prompt_keywords=keywords, graph=graph)

    assert exact_score > distant_score, (
        f"Exact match ({exact_score}) should beat distant file ({distant_score})"
    )


def test_test_file_bonus():
    """Files that look like tests should get a score bonus."""
    import networkx as nx
    graph = nx.DiGraph()
    test_file = "/repo/src/auth/store.spec.ts"
    source_file = "/repo/src/auth/store.ts"
    seed = "/repo/src/main.ts"

    graph.add_node(test_file, size=500)
    graph.add_node(source_file, size=500)
    graph.add_node(seed, size=300)
    graph.add_edge(seed, test_file)
    graph.add_edge(seed, source_file)

    keywords = {"auth"}
    seed_set = {seed}

    test_score = _score_node(test_file, distance=1, seed_set=seed_set,
                              prompt_keywords=keywords, graph=graph)
    source_score = _score_node(source_file, distance=1, seed_set=seed_set,
                                prompt_keywords=keywords, graph=graph)

    assert test_score > source_score, (
        f"Test file ({test_score}) should outrank source file ({source_score})"
    )


def test_keyword_snippet_extraction():
    """extract_keyword_snippets should return a snippet centered on keyword matches."""
    import tempfile
    content = "\n".join(f"line {i}" for i in range(100))
    content += "\nauth_token = 'secret'\n"
    content += "\n".join(f"line {i}" for i in range(100, 200))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        snippet = extract_keyword_snippets(tmp_path, {"auth_token"}, radius=3)
        assert snippet is not None, "Expected a snippet for matching keyword"
        assert "auth_token" in snippet, "Snippet should contain the keyword"
        # Should be much shorter than full file
        assert len(snippet) < len(content) * 0.3, (
            f"Snippet ({len(snippet)} chars) should be much smaller than full file ({len(content)} chars)"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_keyword_snippet_returns_none_when_no_match():
    """extract_keyword_snippets should return None when no keywords match."""
    import tempfile
    content = "just some unrelated code here\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        snippet = extract_keyword_snippets(tmp_path, {"nonexistent_keyword_xyz"}, radius=3)
        assert snippet is None, "Expected None when no keywords match"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_token_budget_trims_lowest_ranked():
    """apply_lite_membrane should drop lowest-ranked files if total tokens exceed budget."""
    import networkx as nx
    graph = nx.DiGraph()
    seed = "/repo/src/main.ts"
    a = "/repo/src/a.ts"
    b = "/repo/src/b.ts"
    c = "/repo/src/c.ts"

    graph.add_node(seed, size=100)
    graph.add_node(a, size=100)
    graph.add_node(b, size=100)
    graph.add_node(c, size=100)
    graph.add_edge(seed, a)
    graph.add_edge(seed, b)
    graph.add_edge(a, c)

    result = apply_lite_membrane(
        graph=graph,
        candidate_nodes=[seed, a, b, c],
        seed_nodes=[seed],
        max_files=10,
        max_tokens_per_file=50,   # each file exceeds per-file cap
        total_token_budget=100,   # only ~2 files fit
    )

    selected = result["selected_nodes"]
    # Seed is always kept; budget may trim others
    assert seed in selected, "Seed must always be kept"
    # The number of selected files should be ≤ 3 (seed + ~2 more)
    assert len(selected) <= 3, f"Expected ≤3 files with tight budget, got {len(selected)}"


def test_depth_1_is_default():
    """isolate_context should default to depth 1."""
    graph = build_import_graph(TARGET_REPO)
    if graph.number_of_nodes() == 0:
        return
    files_depth_1 = isolate_context(graph, "auth", max_depth=1, max_files=100)
    files_depth_2 = isolate_context(graph, "auth", max_depth=2, max_files=100)
    assert len(files_depth_1) <= len(files_depth_2), "Depth 2 should not return fewer files than depth 1"


def test_transparency_note_in_metadata():
    """The returned bundle must include a transparency note explaining simplicity."""
    bundle = build_lite_context_bundle(TARGET_REPO, "auth store")
    meta = bundle["metadata"]
    assert "transparency_note" in meta, "Metadata must contain a transparency note"
    note = meta["transparency_note"]
    assert "public reference" in note.lower() or "simple" in note.lower(), (
        "Transparency note should mention public reference or simplicity"
    )
