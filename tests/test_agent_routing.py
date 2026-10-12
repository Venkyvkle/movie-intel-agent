from movie_agent.agent.agent import SessionState
from movie_agent.agent.router import heuristic_route


def test_heuristic_intents():
    assert heuristic_route("Who said the light never went dark?").intent == "question"
    assert heuristic_route("Email the storm scene breakdown to a@b.com").intent == "email"
    assert heuristic_route("Send the diner scene summary").intent == "email"
    assert heuristic_route("hello").intent == "smalltalk"


def test_question_routes_to_rag_with_citations(agent):
    r = agent.handle("Why did the inquiry blame the lighthouse keeper?", SessionState())
    assert r.kind == "answer" and r.citations
    assert r.citations[0]["title"] == "The Last Lighthouse"
    assert r.citations[0]["start"] and r.citations[0]["best_line_start"]


def test_ambiguous_title_asks(agent):
    s = SessionState()
    r = agent.handle("What happens in Starfall?", s)
    assert r.kind == "clarification" and len(r.options) == 2
    r = agent.handle("2", s)
    assert r.kind == "answer"
    assert {c["title"] for c in r.citations} == {"Starfall Rising"}


def test_unknown_title_asks(agent):
    r = agent.handle("What happens in the Titanic movie?", SessionState())
    assert r.kind == "clarification" and "Titanic" in r.message


def test_ambiguous_quote_asks(agent):
    s = SessionState()
    r = agent.handle('Who said "we\'re not done yet"?', s)
    assert r.kind == "clarification" and len(r.options) == 3
    r = agent.handle("Code of Silence", s)
    assert r.kind == "answer" and r.citations[0]["title"] == "Code of Silence"


def test_email_missing_recipient_then_send(agent):
    s = SessionState()
    r = agent.handle("Email a dialogue summary of the diner confession", s)
    assert r.kind == "clarification" and "email address" in r.message
    r = agent.handle("sam@example.com", s)
    assert r.kind == "email_sent"
    sent = agent.email.sent[-1]
    assert sent["to"] == "sam@example.com" and "Sources" in sent["body"]


def test_email_named_recipient_without_address(agent):
    r = agent.handle("Send a quote analysis of Victor's silence line to Priya", SessionState())
    assert r.kind == "clarification" and "Priya" in r.message


def test_email_previous_answer(agent):
    s = SessionState()
    agent.handle("What does June confess to Detective Hale?", s)
    r = agent.handle("forward that to kim@example.com", s)
    assert r.kind == "email_sent" and any("reusing previous" in t for t in r.trace)
