"""Build the subtitle index from a directory of .srt files.

    python scripts/ingest.py [--dir data/subtitles]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movie_agent.retrieval.builder import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder with up to 100 .srt files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    idx = build_index(Path(args.dir) if args.dir else None)
    print(f"Indexed {len(idx.movies)} movies / {len(idx.chunks)} chunks with {idx.embedder.name} embeddings.")
