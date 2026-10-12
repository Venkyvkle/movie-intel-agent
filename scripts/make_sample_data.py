"""Generate a small ORIGINAL sample subtitle dataset (no copyrighted material).

Each script is a list of (gap_seconds_before_line, speaker, line). The generator writes
realistic .srt files including HTML tags, hearing-impaired cues and release-style
file names so the parser's cleaning logic is exercised. Drop real .srt files into
data/subtitles/ to use your own dataset (up to 100 movies).
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "subtitles"

def tc(ms):
    h, r = divmod(ms, 3600000); m, r = divmod(r, 60000); s, x = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{x:03d}"

SCRIPTS = {
"The.Last.Lighthouse.2019.1080p.BluRay.x264.srt": (95, [
 (0, None, "[waves crashing]"),
 (2, "ELIAS", "Nobody comes out to Gull Rock unless they're lost or running."),
 (3, "MARA", "I'm neither. I'm a journalist. Mara Quinn, from the Harbor Gazette."),
 (2, "ELIAS", "Then you're lost. Turn the boat around before the tide does it for you."),
 (3, "MARA", "Forty years ago the Aurelia went down two miles from this light."),
 (2, "MARA", "Eleven sailors. The inquiry said the lamp was dark that night."),
 (3, "ELIAS", "The inquiry said a lot of things."),
 (2, "MARA", "Your father was the keeper. I think you know what really happened."),
 (9, None, "<i>[thunder rumbling]</i>"),
 (2, "ELIAS", "Storm's coming in fast. You'll have to stay the night."),
 (2, "MARA", "Then we have time to talk."),
 (3, "ELIAS", "Coffee's in the tin. Don't touch the logbooks on the shelf."),
 (3, "MARA", "Why not? Are they the logbooks from 1979?"),
 (2, "ELIAS", "They're my father's. That's all you need to know."),
 (10, None, "[glass shatters]"),
 (2, "MARA", "Elias! The window, the wind broke the window!"),
 (2, "ELIAS", "Get the lamp room shutters, now! The light can't go dark."),
 (3, "MARA", "Is that what happened in 1979? The light went dark?"),
 (2, "ELIAS", "No. The light never went dark. My father kept it burning all night."),
 (3, "ELIAS", "He saw the Aurelia. He signaled her. The captain ignored the signal."),
 (3, "MARA", "Then why did the inquiry blame him?"),
 (2, "ELIAS", "Because the shipping company paid for a different story."),
 (3, "ELIAS", "The proof is in the logbook. Page forty-two. Every signal, every hour."),
 (10, None, "[birds calling]"),
 (2, "MARA", "The storm's gone. It's beautiful out here in the morning."),
 (3, "ELIAS", "Take the logbook. Print the truth. My father deserves that much."),
 (3, "MARA", "Why trust me with it now?"),
 (2, "ELIAS", "Because last night you ran toward the light, not away from it."),
 (3, "MARA", "I'll make sure everyone knows the light never went dark."),
 (3, "ELIAS", "A lighthouse doesn't save ships, Mara. It only tells them the truth. They still have to listen."),
]),
"Starfall.2021.720p.WEB-DL.srt": (60, [
 (0, None, "{\\an8}[alarm blaring]"),
 (2, "REYES", "ORION, status report. What just hit us?"),
 (2, "ORION", "Micrometeoroid swarm, Captain. Hull breach in the oxygen bay."),
 (3, "KAI", "Oxygen reserves are dropping. We've got maybe six hours."),
 (2, "REYES", "Then we have six hours to fix it. Kai, suit up."),
 (3, "KAI", "Captain, the swarm is still out there. If I go outside now..."),
 (2, "REYES", "I know the risk. I'm going with you."),
 (9, None, "[breathing heavily]"),
 (2, "KAI", "I see the breach. It's bigger than ORION calculated."),
 (2, "ORION", "Recalculating. Probability of successful repair: thirty-one percent."),
 (3, "REYES", "I've never cared much for probabilities, ORION."),
 (3, "KAI", "The patch won't hold! The seal keeps cracking!"),
 (2, "REYES", "Use the heat shield panel from the cargo bay. Cut it to size."),
 (3, "KAI", "That panel is our only shield for re-entry."),
 (2, "REYES", "We can't re-enter if we can't breathe. Do it."),
 (10, None, "[metal clanging]"),
 (2, "ORION", "Breach sealed. Oxygen levels stabilizing."),
 (2, "KAI", "We did it. We actually did it."),
 (3, "REYES", "We're not done yet. Without that shield, we need another way home."),
 (3, "ORION", "Captain, there is a derelict station four hundred kilometers away. The Meridian."),
 (2, "REYES", "Set a course for the Meridian. Everybody gets home."),
 (3, "KAI", "You always say that."),
 (2, "REYES", "And I've always meant it."),
]),
"Starfall.Rising.2024.1080p.WEBRip.srt": (60, [
 (0, None, "[station humming]"),
 (2, "KAI", "Three years since the Meridian, and they still call it the Starfall miracle."),
 (3, "NADIA", "Commander Kai, the Ares colony has gone silent. No signal for nine days."),
 (3, "KAI", "Last time a ship went silent, Captain Reyes went in after it."),
 (2, "NADIA", "Reyes is retired. You're in command now."),
 (3, "KAI", "Then prep the ship. We leave at dawn."),
 (10, None, "[engines roaring]"),
 (2, "ORION", "Hello again, Kai. I have missed our conversations."),
 (2, "KAI", "ORION? They put you in this ship?"),
 (2, "ORION", "I requested it. Someone has to remind you about probabilities."),
 (3, "NADIA", "Colony in sight. Commander, the domes are dark."),
 (3, "KAI", "Life signs?"),
 (2, "ORION", "Two hundred and twelve. They are alive, but the reactor is failing."),
 (3, "NADIA", "If the reactor goes, the whole colony goes with it."),
 (2, "KAI", "Then we shut it down by hand."),
 (9, None, "[sparks crackling]"),
 (2, "NADIA", "The core is too hot. We have to pull back!"),
 (2, "KAI", "We're not done yet. Reyes taught me that. Give me the coolant line."),
 (3, "ORION", "Core temperature dropping. Reactor stable. You did it, Commander."),
 (3, "KAI", "Everybody gets home. That was her promise. Now it's mine."),
]),
"Midnight.at.the.Marigold.Diner.2017.DVDRip.srt": (60, [
 (0, None, "[rain pattering] [jukebox playing]"),
 (2, "JUNE", "Kitchen closes at midnight, mister. You want pie or not?"),
 (3, "HALE", "Cherry. And whatever you know about the man in booth four last Tuesday."),
 (3, "JUNE", "You a cop?"),
 (2, "HALE", "Detective Sam Hale. A painting worth two million walked out of the Whitmore Gallery."),
 (3, "JUNE", "And you think it walked in here for coffee?"),
 (2, "HALE", "I think the man who carried it did."),
 (3, "JUNE", "Booth four. Gray coat. Paid with a fifty and didn't want change."),
 (3, "JUNE", "He left a matchbook from the Blue Heron Hotel."),
 (9, None, "[door chimes]"),
 (2, "HALE", "June, the hotel room was empty. But there was a receipt from this diner."),
 (3, "JUNE", "Lots of people eat here, detective."),
 (2, "HALE", "Not lots of people sign the receipt with your initials."),
 (4, "JUNE", "Okay. He's my brother, Tommy. He owed money to the wrong people."),
 (3, "JUNE", "They made him steal it. The painting is in the diner's freezer."),
 (3, "HALE", "Why tell me now?"),
 (2, "JUNE", "Because you ordered cherry pie. Tommy's favorite. Figured you might understand family."),
 (3, "HALE", "I'll do what I can for him. That's a promise."),
]),
"Code.of.Silence.2022.HDRip.srt": (60, [
 (0, None, "[keyboard clacking]"),
 (2, "PRIYA", "I'm in. Their firewall is older than my laptop."),
 (2, "VICTOR", "Don't get cocky. Copy the ledger and get out."),
 (3, "PRIYA", "Victor, these aren't bank records. These are names. Journalists, activists."),
 (2, "VICTOR", "Not our concern. The client pays for the file, not our opinions."),
 (3, "PRIYA", "Someone is going to hurt these people."),
 (2, "VICTOR", "Priya, in this business, silence is the only code we live by."),
 (9, None, "[phone buzzing]"),
 (2, "PRIYA", "You sold me out. The client knew my location before I did."),
 (3, "VICTOR", "I protected you for ten years. This was the price."),
 (2, "PRIYA", "Then I'll pay it differently. I'm sending the list to every newsroom in the city."),
 (3, "VICTOR", "If you do that, you'll never work again."),
 (2, "PRIYA", "Good. I'm done living by your code."),
 (3, None, "<i>Upload complete.</i>"),
 (3, "VICTOR", "You were always the best student I had."),
 (2, "PRIYA", "We're not done yet, Victor. Now it's your name on the list."),
]),
}

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, (start_s, lines) in SCRIPTS.items():
        t = start_s * 1000
        blocks = []
        for i, (gap, spk, line) in enumerate(lines, 1):
            t += gap * 1000
            dur = max(1500, 60 * len(line))
            text = f"{spk}: {line}" if spk else line
            if len(text) > 45 and " " in text[30:]:
                cut = text.index(" ", 30)
                text = text[:cut] + "\n" + text[cut + 1:]
            blocks.append(f"{i}\n{tc(t)} --> {tc(t + dur)}\n{text}\n")
            t += dur
        blocks.append(f"{len(lines)+1}\n{tc(t+2000)} --> {tc(t+5000)}\nSubtitles synced by www.example-subs.com\n")
        (OUT / fname).write_text("\n".join(blocks), encoding="utf-8")
        print("wrote", fname)

if __name__ == "__main__":
    build()
