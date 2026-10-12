from movie_agent.ingestion.metadata import title_from_filename
from movie_agent.ingestion.srt_parser import ms_to_timecode, parse_srt_text
from pathlib import Path

SAMPLE = """\ufeff1
00:00:01,500 --> 00:00:03,250
<i>JOHN: Hello there.</i>

2
00:00:04.000 --> 00:00:06,9
{\\an8}[door slams]
- Who's there?
- It's me.

3
00:00:07,000 --> 00:00:08,000
[MUSIC PLAYING]

4
00:00:09,000 --> 00:00:10,000
Subtitles by www.opensubtitles.org
"""


def test_timecodes_and_cleaning():
    cues = parse_srt_text(SAMPLE.replace("\n", "\r\n"))
    assert len(cues) == 2                      # sound-only and ad cues dropped
    assert cues[0].start_ms == 1500 and cues[0].end_ms == 3250
    assert cues[0].speaker == "John" and cues[0].text == "Hello there."
    assert cues[1].start_ms == 4000 and cues[1].end_ms == 6900   # '.' separator + short ms
    assert cues[1].text == "Who's there? It's me."
    assert cues[1].sound_cues == ["door slams"]


def test_timecode_format():
    assert ms_to_timecode(3_723_456) == "01:02:03"
    assert ms_to_timecode(3_723_456, with_ms=True) == "01:02:03,456"


def test_title_from_filename():
    assert title_from_filename(Path("The.Last.Lighthouse.2019.1080p.BluRay.x264.srt")) == ("The Last Lighthouse", 2019)
    assert title_from_filename(Path("Starfall Rising (2024) [WEBRip].srt")) == ("Starfall Rising", 2024)
