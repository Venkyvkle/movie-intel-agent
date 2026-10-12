from movie_agent.ingestion.chunker import chunk_cues
from movie_agent.ingestion.srt_parser import Cue


def _cues(n, gap_at=None):
    out, t = [], 0
    for i in range(n):
        t += 10_000 if i == gap_at else 500
        out.append(Cue(i + 1, t, t + 2000, f"line number {i}", f"line number {i}"))
        t += 2000
    return out


def test_scene_gap_splits_and_metadata():
    chunks = chunk_cues(_cues(10, gap_at=5), "m", "Movie", 2000, scene_gap_seconds=6)
    assert len(chunks) == 2 and chunks[1].scene_index == 1
    assert chunks[0].cue_start == 1 and chunks[1].cue_start == 6
    assert chunks[0].citation.startswith("Movie (2000), 00:00:00")


def test_overlap_when_budget_exceeded():
    chunks = chunk_cues(_cues(30), "m", "Movie", None, max_chars=120, overlap_cues=2)
    assert len(chunks) > 2
    assert chunks[1].cue_start == chunks[0].cue_end - 1   # 2-cue overlap


def test_retrieval_finds_right_movie(index):
    hits = index.search("painting hidden in the freezer", k=3)
    assert hits[0].chunk.title == "Midnight at the Marigold Diner"
    hits = index.search("oxygen bay hull breach", k=3, movie_ids=["starfall-2021"])
    assert all(h.chunk.movie_id == "starfall-2021" for h in hits)


def test_index_metadata(index):
    assert len(index.movies) == 5
    c = index.chunks[0]
    assert c.lines and {"start", "end", "text"} <= set(c.lines[0])
