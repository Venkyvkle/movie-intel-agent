"""Terminal chat with the agent:  python scripts/cli.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from movie_agent.agent.agent import MovieAgent, SessionState  # noqa: E402
from movie_agent.retrieval.builder import load_or_build  # noqa: E402

if __name__ == "__main__":
    agent, state = MovieAgent(load_or_build()), SessionState()
    print(f"Loaded {len(agent.index.movies)} movies. LLM: {agent.llm.provider}. Type 'quit' to exit.")
    while (msg := input("\nyou> ").strip()).lower() not in {"quit", "exit"}:
        r = agent.handle(msg, state)
        print(f"\n[{r.kind}] {r.message}")
        for c in r.citations:
            print(f"   · {c['label']}  (key line {c['best_line_start']}: \"{c['best_line'][:70]}\")")
        print("   route:", " | ".join(r.trace))
