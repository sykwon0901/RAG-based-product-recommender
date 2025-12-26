# scripts/extract_functions_to_src.py
# Extract top-level functions from a notebook into src modules (fast refactor step)

from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Dict, List

NOTEBOOK_PATH = "notebooks/01_pipeline_clean.ipynb"
PKG_DIR = Path("src/amazon_skincare_graphrag")

# Map function name -> module filename
FUNC_TO_MODULE: Dict[str, str] = {
    # neo4j io / schema
    "init_neo4j_schema": "neo4j_io.py",
    "import_products": "neo4j_io.py",
    "import_reviews": "neo4j_io.py",
    "get_reviews_without_embeddings": "neo4j_io.py",
    "update_review_embeddings": "neo4j_io.py",

    # embeddings / vector index
    "get_embedding": "embeddings.py",
    "generate_embeddings": "embeddings.py",
    "create_vector_index": "embeddings.py",

    # retrieval / personalization
    "run_search": "retrieval.py",
    "run_hybrid_query": "retrieval.py",
    "analyze_user_loyalty": "retrieval.py",
    "get_user_last_review_text": "retrieval.py",

    # llm helpers
    "generate_synthetic_query": "llm.py",
    "generate_answer": "llm.py",
    "generate_ragas_answer": "llm.py",

    # evaluation
    "fetch_test_dataset": "eval.py",
    "calculate_metrics": "eval.py",
    "get_existing_col": "eval.py",
}

HEADER = """\
# Auto-generated from notebooks/01_pipeline_clean.ipynb
# You can manually refine imports and signatures later.

from __future__ import annotations
"""

def read_notebook_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            chunks.append("".join(cell.get("source", [])))
    return "\n\n".join(chunks)

def extract_import_lines(code: str) -> List[str]:
    # Keep only top-level imports (simple heuristic using AST)
    tree = ast.parse(code)
    import_lines = []
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = "".join(lines[node.lineno - 1 : node.end_lineno])
            import_lines.append(seg)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in import_lines:
        key = s.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out

def extract_function_source(code: str) -> Dict[str, str]:
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    out: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name in FUNC_TO_MODULE:
                seg = "".join(lines[node.lineno - 1 : node.end_lineno])
                out[name] = seg
    return out

def main() -> None:
    nb_path = Path(NOTEBOOK_PATH)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    PKG_DIR.mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "__init__.py").write_text("# Package init\n", encoding="utf-8")

    code = read_notebook_code(nb_path)
    imports = extract_import_lines(code)
    funcs = extract_function_source(code)

    # Group functions by module
    module_to_funcs: Dict[str, List[str]] = {}
    for fn, mod in FUNC_TO_MODULE.items():
        if fn in funcs:
            module_to_funcs.setdefault(mod, []).append(fn)

    # Write modules
    for mod, fn_names in module_to_funcs.items():
        path = PKG_DIR / mod
        content = [HEADER]
        if imports:
            content.append("# Imports extracted from the notebook\n")
            content.extend(imports)
            content.append("\n")
        content.append("# Functions extracted from the notebook\n\n")
        for fn in fn_names:
            content.append(funcs[fn])
            if not funcs[fn].endswith("\n"):
                content.append("\n")
            content.append("\n\n")
        path.write_text("".join(content), encoding="utf-8")

    missing = [fn for fn in FUNC_TO_MODULE.keys() if fn not in funcs]
    print("[Done] Generated modules under src/amazon_skincare_graphrag/")
    if missing:
        print("[Warning] Functions not found in the notebook (skipped):")
        for fn in missing:
            print(" -", fn)

if __name__ == "__main__":
    main()
