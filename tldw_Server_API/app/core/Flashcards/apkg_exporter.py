# apkg_exporter.py
#
# Imports
import io
import json
import mimetypes
import os
import sqlite3
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from tldw_Server_API.app.core.Flashcards.asset_refs import replace_markdown_asset_refs_for_export

#
########################################################################################################################
#
# Functions:

def _now_secs() -> int:
    return int(time.time())


def _now_millis() -> int:
    return int(time.time() * 1000)


def _day_start_secs(dt: Optional[datetime] = None) -> int:
    dt = dt or datetime.now(timezone.utc)
    day = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(day.timestamp())


def _sha1_8_int(s: str) -> int:
    import hashlib
    h = hashlib.sha1(s.encode('utf-8'), usedforsecurity=False).hexdigest()
    return int(h[:8], 16)


def _guess_extension(mime_type: str | None, original_filename: str | None = None) -> str:
    if original_filename:
        ext = os.path.splitext(original_filename)[1].lstrip(".").lower()
        if ext:
            return "jpg" if ext == "jpeg" else ext
    guessed = mimetypes.guess_extension(str(mime_type or "").strip().lower()) or ""
    ext = guessed.lstrip(".").lower()
    if ext:
        return "jpg" if ext == "jpeg" else ext
    return "bin"


def _build_models_json(basic_mid: int, cloze_mid: int) -> dict:
    # Basic model
    basic = {
        "css": ".card { font-family: arial; font-size: 20px; text-align: left; color: black; background-color: white; }",
        "did": None,
        "flds": [
            {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "media": []},
            {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "media": []},
            {"name": "Extra", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 16, "media": []},
            {"name": "Notes", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 16, "media": []},
        ],
        "id": basic_mid,
        "latexPost": "\\end{document}",
        "latexPre": "\\documentclass[12pt]{article}\\special{papersize=3in,5in}\\usepackage[utf8]{inputenc}\\usepackage{amssymb,amsmath}\\pagestyle{empty}\\setlength{\\parindent}{0in}\\begin{document}",
        "mod": _now_secs(),
        "name": "Basic",
        "req": [],
        "sortf": 0,
        "tags": [],
        "tmpls": [
            {
                "name": "Card 1",
                "ord": 0,
                "qfmt": "{{Front}}",
                "afmt": "{{Front}}<hr id=answer>{{Back}}<br>{{Extra}}",
                "bafmt": "",
                "bqfmt": "",
                "did": None,
            },
            {
                "name": "Card 2",
                "ord": 1,
                "qfmt": "{{Back}}",
                "afmt": "{{Back}}<hr id=answer>{{Front}}<br>{{Extra}}",
                "bafmt": "",
                "bqfmt": "",
                "did": None,
            }
        ],
        "type": 0,
        "usn": -1,
        "vers": [],
    }

    cloze = {
        "css": basic["css"],
        "did": None,
        "flds": [
            {"name": "Text", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "media": []},
            {"name": "Back Extra", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 16, "media": []},
            {"name": "Notes", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 16, "media": []},
        ],
        "id": cloze_mid,
        "latexPost": basic["latexPost"],
        "latexPre": basic["latexPre"],
        "mod": _now_secs(),
        "name": "Cloze",
        "req": [],
        "sortf": 0,
        "tags": [],
        "tmpls": [
            {
                "name": "Cloze",
                "ord": 0,
                "qfmt": "{{cloze:Text}}",
                "afmt": "{{cloze:Text}}<br>{{Back Extra}}",
                "bafmt": "",
                "bqfmt": "",
                "did": None,
            }
        ],
        "type": 1,
        "usn": -1,
        "vers": [],
    }

    return {str(basic_mid): basic, str(cloze_mid): cloze}


def _build_decks_json(deck_map: dict[int, str]) -> dict:
    now = _now_secs()
    decks = {}
    for deck_id, name in deck_map.items():
        decks[str(deck_id)] = {
            "name": name or "Default",
            "usn": -1,
            "collapsed": False,
            "browserCollapsed": False,
            "newToday": [0, 0],
            "revToday": [0, 0],
            "lrnToday": [0, 0],
            "timeToday": [0, 0],
            "dyn": 0,
            "extendNew": 10,
            "extendRev": 10,
            "conf": 1,
            "id": deck_id,
            "mod": now,
            "desc": "",
        }
    return decks


def _build_dconf_json() -> dict:
    now = _now_secs()
    return {
        "1": {
            "autoplay": True,
            "dyn": 0,
            "id": 1,
            "lapse": {"delays": [10], "leechAction": 0, "leechFails": 8, "minInt": 1, "mult": 0.5},
            "maxTaken": 60,
            "mod": now,
            "name": "Default",
            "new": {
                "bury": True,
                "delays": [1, 10],
                "initialFactor": 2500,
                "ints": [1, 4, 0],
                "order": 0,
                "perDay": 20,
                "separate": True,
            },
            "replayq": True,
            "rev": {
                "bury": True,
                "ease4": 1.3,
                "fuzz": 0.05,
                "ivlFct": 1,
                "maxIvl": 36500,
                "minSpace": 1,
                "perDay": 200,
            },
            "timer": 1,
            "usn": -1,
        }
    }


def _build_conf_json(default_deck_id: int) -> dict:
    return {
        "curDeck": default_deck_id,
        "activeDecks": [default_deck_id],
        "newSpread": 0,
        "collapseTime": 1800,
        "timeLim": 3600,
        "estTimes": True,
        "dueCounts": True,
        "curModel": "",
        "nextPos": 1,
        "sortType": "noteFld",
        "sortBackwards": False,
        "addToCur": True,
        "dayLearnFirst": False,
        "newBury": True,
        "activeCols": ["noteFld", "template", "cardDue", "deck"],
    }


def _compute_card_sched(model_type: str, ef: float, interval_days: int, repetitions: int, lapses: int,
                        due_at_iso: Optional[str], col_crt_secs: int) -> tuple[int, int, int, int, int, int, int]:
    # Returns: (type, queue, due, ivl, factor, reps, lapses)
    factor = int(round(ef * 1000)) if ef else 2500
    reps_val = int(repetitions or 0)
    lapses_val = int(lapses or 0)
    # New: never studied yet
    if reps_val == 0:
        return (0, 0, 0, 0, factor, reps_val, lapses_val)

    # Learning: studied at least once, no interval yet
    if interval_days == 0 or reps_val in (1, 2):
        # Learning
        due_secs = _now_secs()
        if due_at_iso:
            try:
                due_dt = datetime.fromisoformat(due_at_iso.replace('Z', '+00:00'))
                due_secs = int(due_dt.timestamp())
            except Exception as due_parse_error:
                logger.debug("APKG exporter failed to parse due_at for new-card note", exc_info=due_parse_error)
        return (1, 1, due_secs, 0, factor, reps_val, lapses_val)
    # Review
    ivl_days = max(1, int(interval_days))
    # due: days since collection creation (prefer due_at when available)
    days_since_crt = max(1, int((datetime.now(timezone.utc).timestamp() - col_crt_secs) / 86400.0))
    if due_at_iso:
        try:
            due_dt = datetime.fromisoformat(due_at_iso.replace('Z', '+00:00'))
            days_since_crt = max(1, int((due_dt.timestamp() - col_crt_secs) / 86400.0))
        except Exception as due_parse_error:
            logger.debug("APKG exporter failed to parse due_at for review-card note", exc_info=due_parse_error)
    return (2, 2, days_since_crt, ivl_days, factor, reps_val, lapses_val)


import re


def _extract_media_from_html(
    html: str,
    media_accum: list[tuple[str, bytes]],
    media_map: dict[str, int],
    *,
    media_total_bytes: list[int] | None = None,
    max_total_media_bytes: int | None = None,
) -> str:
    """
    Extract data URIs in img/audio tags, store as files, and replace src with filename.
    media_accum: list of (filename, bytes) to be written later
    media_map: filename -> index assigned
    Returns modified HTML.
    """
    def repl(match):
        tag = match.group(0)
        src = match.group(1)
        if not src.startswith('data:'):
            return tag
        # data URI: data:<mime>;base64,<data>
        try:
            header, b64data = src.split(',', 1)
            mime = header.split(';')[0][5:]
            ext = 'bin'
            if '/' in mime:
                ext = mime.split('/')[-1].split('+')[0]
                if ext == 'jpeg':
                    ext = 'jpg'
            import base64
            content = base64.b64decode(b64data)
            if media_total_bytes is not None:
                media_total_bytes[0] += len(content)
                if max_total_media_bytes is not None and media_total_bytes[0] > max_total_media_bytes:
                    raise ValueError(f"APKG media exceeds max total size of {max_total_media_bytes} bytes")
            # filename
            fname_base = f"media_{uuid.uuid4().hex[:8]}.{ext}"
            # avoid collision
            filename = fname_base
            idx = len(media_accum)
            media_map[filename] = idx
            media_accum.append((filename, content))
            return tag.replace(src, filename)
        except Exception:
            return tag

    # Replace img and audio src
    re.compile(r'(?:<img[^>]+src\s*=\s*["\"])([^"\"]+)(?:["\"][^>]*>)|(?:<audio[^>]+src\s*=\s*["\"])([^"\"]+)(?:["\"][^>]*>)', re.IGNORECASE)
    # We need to handle both capturing groups; easier: custom parser for img/src and audio/src separately
    def replace_tag_src(tag_name: str, s: str) -> str:
        # Match src with either single or double quotes
        rgx = re.compile(rf'<{tag_name}[^>]*?\s+src\s*=\s*([\'"\"])\s*(.*?)\1', re.IGNORECASE)
        return rgx.sub(lambda m: m.group(0).replace(m.group(2), _handle_src(m.group(2))), s)

    def _handle_src(src: str) -> str:
        if not src.startswith('data:'):
            return src
        try:
            header, b64data = src.split(',', 1)
            mime = header.split(';')[0][5:]
            ext = 'bin'
            if '/' in mime:
                ext = mime.split('/')[-1].split('+')[0]
                if ext == 'jpeg':
                    ext = 'jpg'
            import base64
            content = base64.b64decode(b64data)
            if media_total_bytes is not None:
                media_total_bytes[0] += len(content)
                if max_total_media_bytes is not None and media_total_bytes[0] > max_total_media_bytes:
                    raise ValueError(f"APKG media exceeds max total size of {max_total_media_bytes} bytes")
            filename = f"media_{uuid.uuid4().hex[:8]}.{ext}"
            idx = len(media_accum)
            media_map[filename] = idx
            media_accum.append((filename, content))
            return filename
        except Exception:
            return src

    html = replace_tag_src('img', html)
    html = replace_tag_src('audio', html)
    return html


def export_apkg_from_rows(
    rows: list[dict],
    default_deck_name: str = "Default",
    include_reverse: bool = False,
    *,
    asset_loader=None,
    max_total_media_bytes: int | None = None,
) -> bytes:
    """
    Build an APKG bytes object from flashcard rows returned by list_flashcards().
    Each row should contain: deck_name, front, back, notes, extra, model_type, ef, interval_days, repetitions, lapses, due_at.
    """
    # Prepare deck ids
    unique_decks = sorted({r.get("deck_name") or default_deck_name for r in rows})
    base_id = _now_millis()
    deck_ids: dict[str, int] = {name: base_id + i for i, name in enumerate(unique_decks)}

    # Model ids
    basic_mid = base_id + 100000
    cloze_mid = base_id + 100001

    # Open temp sqlite
    with tempfile.TemporaryDirectory() as tmp:
        col_path = os.path.join(tmp, "collection.anki2")
        conn = sqlite3.connect(col_path)
        c = conn.cursor()
        # Create schema
        c.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS col(
              id integer primary key,
              crt integer not null,
              mod integer not null,
              scm integer not null,
              ver integer not null,
              dty integer not null,
              usn integer not null,
              ls integer not null,
              conf text not null,
              models text not null,
              decks text not null,
              dconf text not null,
              tags text not null
            );
            CREATE TABLE IF NOT EXISTS notes(
              id integer primary key,
              guid text not null,
              mid integer not null,
              mod integer not null,
              usn integer not null,
              tags text not null,
              flds text not null,
              sfld integer not null,
              csum integer not null,
              flags integer not null,
              data text not null
            );
            CREATE TABLE IF NOT EXISTS cards(
              id integer primary key,
              nid integer not null,
              did integer not null,
              ord integer not null,
              mod integer not null,
              usn integer not null,
              type integer not null,
              queue integer not null,
              due integer not null,
              ivl integer not null,
              factor integer not null,
              reps integer not null,
              lapses integer not null,
              left integer not null,
              odue integer not null,
              odid integer not null,
              flags integer not null,
              data text not null
            );
            CREATE TABLE IF NOT EXISTS revlog(
              id integer primary key,
              cid integer not null,
              usn integer not null,
              ease integer not null,
              ivl integer not null,
              lastIvl integer not null,
              factor integer not null,
              time integer not null,
              type integer not null
            );
            CREATE TABLE IF NOT EXISTS graves(
              usn integer not null,
              oid integer not null,
              type integer not null
            );
            """
        )

        col_id = 1
        crt = _day_start_secs()
        now_ms = _now_millis()
        models_json = _build_models_json(basic_mid, cloze_mid)
        decks_json = _build_decks_json({deck_ids[name]: name for name in unique_decks})
        dconf_json = _build_dconf_json()
        conf_json = _build_conf_json(next(iter(deck_ids.values())))
        tags_json = {}
        c.execute(
            "INSERT INTO col(id,crt,mod,scm,ver,dty,usn,ls,conf,models,decks,dconf,tags) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                col_id, crt, now_ms, now_ms, 11, 0, 0, now_ms,
                json.dumps(conf_json), json.dumps(models_json), json.dumps(decks_json), json.dumps(dconf_json), json.dumps(tags_json)
            )
        )

        # Insert notes + cards
        next_new_pos = 1
        nid_base = now_ms + 200000
        cid_base = now_ms + 300000
        card_seq = 0
        media_accum: list[tuple[str, bytes]] = []
        media_idx_map: dict[str, int] = {}
        media_total_bytes = [0]
        managed_asset_filenames: dict[str, str] = {}

        def _register_managed_asset(asset_uuid: str) -> str:
            if asset_uuid in managed_asset_filenames:
                return managed_asset_filenames[asset_uuid]
            if asset_loader is None:
                raise ValueError(f"Managed flashcard asset export is unavailable for {asset_uuid}")

            loaded = asset_loader(asset_uuid)
            if isinstance(loaded, dict):
                content = loaded.get("content") or loaded.get("bytes")
                mime_type = loaded.get("mime_type")
                original_filename = loaded.get("original_filename")
            elif isinstance(loaded, (tuple, list)) and len(loaded) >= 2:
                content = loaded[0]
                mime_type = loaded[1]
                original_filename = loaded[2] if len(loaded) > 2 else None
            else:
                raise ValueError(f"Unsupported asset loader payload for {asset_uuid}")

            if not isinstance(content, (bytes, bytearray, memoryview)):
                raise ValueError(f"Managed flashcard asset {asset_uuid} is missing bytes")
            content_bytes = bytes(content)
            media_total_bytes[0] += len(content_bytes)
            if max_total_media_bytes is not None and media_total_bytes[0] > max_total_media_bytes:
                raise ValueError(f"APKG media exceeds max total size of {max_total_media_bytes} bytes")

            filename = f"managed_{asset_uuid.replace('-', '')[:12]}.{_guess_extension(mime_type, original_filename)}"
            media_idx_map[filename] = len(media_accum)
            media_accum.append((filename, content_bytes))
            managed_asset_filenames[asset_uuid] = filename
            return filename

        for i, r in enumerate(rows):
            deck_name = r.get("deck_name") or default_deck_name
            did = deck_ids[deck_name]
            model_type = r.get("model_type") or ("cloze" if r.get("is_cloze") else "basic")
            is_cloze = (model_type == 'cloze')
            reverse_flag = bool(r.get("reverse"))
            ef = float(r.get("ef") or 2.5)
            interval_days = int(r.get("interval_days") or 0)
            repetitions = int(r.get("repetitions") or 0)
            lapses = int(r.get("lapses") or 0)
            due_at = r.get("due_at")

            front = r.get("front") or ""
            back = r.get("back") or ""
            notes = r.get("notes") or ""
            extra = r.get("extra") or ""
            if asset_loader is not None:
                front = replace_markdown_asset_refs_for_export(front, _register_managed_asset)
                back = replace_markdown_asset_refs_for_export(back, _register_managed_asset)
                extra = replace_markdown_asset_refs_for_export(extra, _register_managed_asset)
                notes = replace_markdown_asset_refs_for_export(notes, _register_managed_asset)
            # Extract media from HTML fields and replace with filenames
            front = _extract_media_from_html(
                front,
                media_accum,
                media_idx_map,
                media_total_bytes=media_total_bytes,
                max_total_media_bytes=max_total_media_bytes,
            )
            back = _extract_media_from_html(
                back,
                media_accum,
                media_idx_map,
                media_total_bytes=media_total_bytes,
                max_total_media_bytes=max_total_media_bytes,
            )
            extra = _extract_media_from_html(
                extra,
                media_accum,
                media_idx_map,
                media_total_bytes=media_total_bytes,
                max_total_media_bytes=max_total_media_bytes,
            )
            notes = _extract_media_from_html(
                notes,
                media_accum,
                media_idx_map,
                media_total_bytes=media_total_bytes,
                max_total_media_bytes=max_total_media_bytes,
            )
            tags_json_str = r.get("tags_json")
            tags_list = []
            if tags_json_str:
                try:
                    tags_list = json.loads(tags_json_str)
                except Exception:
                    tags_list = []
            tags_str = " " + " ".join(t for t in tags_list) + " " if tags_list else ""

            if is_cloze:
                mid = cloze_mid
                flds = f"{front}\x1f{extra}\x1f{notes}"
                sfld = 0
            else:
                mid = basic_mid
                flds = f"{front}\x1f{back}\x1f{extra}\x1f{notes}"
                sfld = 0

            nid = nid_base + i
            guid = uuid.uuid4().hex[:10]
            csum = _sha1_8_int(front)
            c.execute(
                "INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (nid, guid, mid, _now_secs(), -1, tags_str, flds, sfld, csum, 0, "")
            )

            if is_cloze:
                # Generate one card per unique cloze index N in Text
                cloze_ids = {int(m.group(1)) for m in re.finditer(r"\{\{c(\d+)::", front)}
                if not cloze_ids:
                    cloze_ids = {1}
                for n in sorted(cloze_ids):
                    type_v, queue_v, due_v, ivl_v, factor_v, reps_v, lapses_v = _compute_card_sched(model_type, ef, interval_days, repetitions, lapses, due_at, crt)
                    if type_v == 0:
                        due_v = next_new_pos
                        next_new_pos += 1
                    cid = cid_base + card_seq
                    card_seq += 1
                    ord_n = max(0, n - 1)
                    c.execute(
                        "INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,left,odue,odid,flags,data) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (cid, nid, did, ord_n, _now_secs(), -1, type_v, queue_v, due_v, ivl_v, factor_v, reps_v, lapses_v, 0, 0, 0, 0, "")
                    )
            else:
                # Card 1
                type_v, queue_v, due_v, ivl_v, factor_v, reps_v, lapses_v = _compute_card_sched(model_type, ef, interval_days, repetitions, lapses, due_at, crt)
                if type_v == 0:
                    due_v = next_new_pos
                    next_new_pos += 1
                cid = cid_base + card_seq
                card_seq += 1
                c.execute(
                    "INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,left,odue,odid,flags,data) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, nid, did, 0, _now_secs(), -1, type_v, queue_v, due_v, ivl_v, factor_v, reps_v, lapses_v, 0, 0, 0, 0, "")
                )
                # Reverse card if requested/model says so
                if reverse_flag or model_type == 'basic_reverse' or include_reverse:
                    cid2 = cid_base + card_seq
                    card_seq += 1
                    c.execute(
                        "INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,left,odue,odid,flags,data) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (cid2, nid, did, 1, _now_secs(), -1, type_v, queue_v, due_v, ivl_v, factor_v, reps_v, lapses_v, 0, 0, 0, 0, "")
                    )

        conn.commit()
        conn.close()

        # Package zip
        apkg_bytes = io.BytesIO()
        with zipfile.ZipFile(apkg_bytes, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(col_path, arcname='collection.anki2')
            # Build media mapping: index string -> filename
            media_mapping = {str(idx): fname for idx, (fname, _) in enumerate(media_accum)}
            for idx, (_fname, content) in enumerate(media_accum):
                z.writestr(str(idx), content)
            z.writestr('media', json.dumps(media_mapping))
        return apkg_bytes.getvalue()

#
# End of apkg_exporter.py
########################################################################################################################
