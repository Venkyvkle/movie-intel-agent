import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    from movie_agent.retrieval.builder import build_index
    return build_index(ROOT / "data" / "subtitles", tmp_path_factory.mktemp("idx"))


class FakeEmail:
    def __init__(self):
        self.sent = []

    def send_email(self, to, subject, body, cc=None):
        self.sent.append({"to": to, "subject": subject, "body": body})
        return {"status": "dry_run", "message_id": "<x>", "to": [to]}


@pytest.fixture()
def agent(index):
    from movie_agent.agent.agent import MovieAgent
    from movie_agent.agent.llm import LLM
    return MovieAgent(index, LLM("none"), FakeEmail())
