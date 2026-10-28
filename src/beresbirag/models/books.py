"""Canonical 66-book registry and abbreviation resolver for the Berean Standard Bible (BSB)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TestamentType = Literal["OT", "NT"]


@dataclass(frozen=True, slots=True)
class BookInfo:
    """Canonical metadata for a BSB Bible book."""

    code: str
    name: str
    short_abbr: str
    testament: TestamentType
    canonical_order: int
    aliases: tuple[str, ...] = ()


BSB_BOOKS: tuple[BookInfo, ...] = (
    # Old Testament (39 Books)
    BookInfo("GEN", "Genesis", "Gen", "OT", 1, ("Ge", "Gn")),
    BookInfo("EXO", "Exodus", "Exo", "OT", 2, ("Ex", "Exod")),
    BookInfo("LEV", "Leviticus", "Lev", "OT", 3, ("Le", "Lv")),
    BookInfo("NUM", "Numbers", "Num", "OT", 4, ("Nu", "Nm", "Nb")),
    BookInfo("DEU", "Deuteronomy", "Deu", "OT", 5, ("Deut", "Dt", "De")),
    BookInfo("JOS", "Joshua", "Jos", "OT", 6, ("Josh", "Jsh")),
    BookInfo("JDG", "Judges", "Jdg", "OT", 7, ("Judg", "Jg", "Jdgs")),
    BookInfo("RUT", "Ruth", "Rut", "OT", 8, ("Rth", "Ru")),
    BookInfo("1SA", "1 Samuel", "1Sa", "OT", 9, ("1 Sam", "1Sam", "1 Sm", "1S", "I Samuel", "I Sam")),
    BookInfo("2SA", "2 Samuel", "2Sa", "OT", 10, ("2 Sam", "2Sam", "2 Sm", "2S", "II Samuel", "II Sam")),
    BookInfo("1KI", "1 Kings", "1Ki", "OT", 11, ("1 Kgs", "1Kgs", "1 Kin", "1Kin", "1K", "I Kings", "I Kgs")),
    BookInfo("2KI", "2 Kings", "2Ki", "OT", 12, ("2 Kgs", "2Kgs", "2 Kin", "2Kin", "2K", "II Kings", "II Kgs")),
    BookInfo("1CH", "1 Chronicles", "1Ch", "OT", 13, ("1 Chr", "1Chr", "1 Chron", "1Chron", "I Chronicles", "I Chr")),
    BookInfo("2CH", "2 Chronicles", "2Ch", "OT", 14, ("2 Chr", "2Chr", "2 Chron", "2Chron", "II Chronicles", "II Chr")),
    BookInfo("EZR", "Ezra", "Ezr", "OT", 15, ("Ez",)),
    BookInfo("NEH", "Nehemiah", "Neh", "OT", 16, ("Ne",)),
    BookInfo("EST", "Esther", "Est", "OT", 17, ("Esth", "Es")),
    BookInfo("JOB", "Job", "Job", "OT", 18, ("Jb",)),
    BookInfo("PSA", "Psalms", "Psa", "OT", 19, ("Psalm", "Ps", "Pss", "Pslm")),
    BookInfo("PRO", "Proverbs", "Pro", "OT", 20, ("Prov", "Prv", "Pr")),
    BookInfo("ECC", "Ecclesiastes", "Ecc", "OT", 21, ("Eccl", "Eccles", "Ec", "Qoh")),
    BookInfo("SNG", "Song of Solomon", "Sng", "OT", 22, ("Song", "Song of Songs", "SOS", "SoS", "Cant", "Canticles")),
    BookInfo("ISA", "Isaiah", "Isa", "OT", 23, ("Is",)),
    BookInfo("JER", "Jeremiah", "Jer", "OT", 24, ("Je", "Jr")),
    BookInfo("LAM", "Lamentations", "Lam", "OT", 25, ("La",)),
    BookInfo("EZK", "Ezekiel", "Ezk", "OT", 26, ("Eze", "Ezek")),
    BookInfo("DAN", "Daniel", "Dan", "OT", 27, ("Da", "Dn")),
    BookInfo("HOS", "Hosea", "Hos", "OT", 28, ("Ho",)),
    BookInfo("JOL", "Joel", "Jol", "OT", 29, ("Joe", "Jl")),
    BookInfo("AMO", "Amos", "Amo", "OT", 30, ("Am",)),
    BookInfo("OBA", "Obadiah", "Oba", "OT", 31, ("Obad", "Ob")),
    BookInfo("JON", "Jonah", "Jon", "OT", 32, ("Jnh",)),
    BookInfo("MIC", "Micah", "Mic", "OT", 33, ("Mi",)),
    BookInfo("NAM", "Nahum", "Nam", "OT", 34, ("Nah", "Na")),
    BookInfo("HAB", "Habakkuk", "Hab", "OT", 35, ("Hb",)),
    BookInfo("ZEP", "Zephaniah", "Zep", "OT", 36, ("Zeph", "Zp")),
    BookInfo("HAG", "Haggai", "Hag", "OT", 37, ("Hg",)),
    BookInfo("ZEC", "Zechariah", "Zec", "OT", 38, ("Zech", "Zc")),
    BookInfo("MAL", "Malachi", "Mal", "OT", 39, ("Ml",)),
    # New Testament (27 Books)
    BookInfo("MAT", "Matthew", "Mat", "NT", 40, ("Matt", "Mt")),
    BookInfo("MRK", "Mark", "Mrk", "NT", 41, ("Mar", "Mk", "Mr")),
    BookInfo("LUK", "Luke", "Luk", "NT", 42, ("Lk", "Lu")),
    BookInfo("JHN", "John", "Jhn", "NT", 43, ("Joh", "Jn")),
    BookInfo("ACT", "Acts", "Act", "NT", 44, ("Ac", "Acts of the Apostles")),
    BookInfo("ROM", "Romans", "Rom", "NT", 45, ("Ro", "Rm")),
    BookInfo("1CO", "1 Corinthians", "1Co", "NT", 46, ("1 Cor", "1Cor", "I Corinthians", "I Cor")),
    BookInfo("2CO", "2 Corinthians", "2Co", "NT", 47, ("2 Cor", "2Cor", "II Corinthians", "II Cor")),
    BookInfo("GAL", "Galatians", "Gal", "NT", 48, ("Ga",)),
    BookInfo("EPH", "Ephesians", "Eph", "NT", 49, ("Ep",)),
    BookInfo("PHP", "Philippians", "Php", "NT", 50, ("Phil", "Pp")),
    BookInfo("COL", "Colossians", "Col", "NT", 51, ("Co",)),
    BookInfo("1TH", "1 Thessalonians", "1Th", "NT", 52, ("1 Thess", "1Thess", "1 Th", "I Thessalonians")),
    BookInfo("2TH", "2 Thessalonians", "2Th", "NT", 53, ("2 Thess", "2Thess", "2 Th", "II Thessalonians")),
    BookInfo("1TI", "1 Timothy", "1Ti", "NT", 54, ("1 Tim", "1Tim", "1 Ti", "I Timothy")),
    BookInfo("2TI", "2 Timothy", "2Ti", "NT", 55, ("2 Tim", "2Tim", "2 Ti", "II Timothy")),
    BookInfo("TIT", "Titus", "Tit", "NT", 56, ("Ti",)),
    BookInfo("PHM", "Philemon", "Phm", "NT", 57, ("Phlm", "Philem", "Pm")),
    BookInfo("HEB", "Hebrews", "Heb", "NT", 58, ("He",)),
    BookInfo("JAS", "James", "Jas", "NT", 59, ("Jam", "Jm")),
    BookInfo("1PE", "1 Peter", "1Pe", "NT", 60, ("1 Pet", "1Pet", "1 Pt", "1Pt", "I Peter")),
    BookInfo("2PE", "2 Peter", "2Pe", "NT", 61, ("2 Pet", "2Pet", "2 Pt", "2Pt", "II Peter")),
    BookInfo("1JN", "1 John", "1Jn", "NT", 62, ("1 Jo", "1Jo", "1 Joh", "1Joh", "1John", "I John")),
    BookInfo("2JN", "2 John", "2Jn", "NT", 63, ("2 Jo", "2Jo", "2 Joh", "2Joh", "2John", "II John")),
    BookInfo("3JN", "3 John", "3Jn", "NT", 64, ("3 Jo", "3Jo", "3 Joh", "3Joh", "3John", "III John")),
    BookInfo("JUD", "Jude", "Jud", "NT", 65, ("Jd",)),
    BookInfo("REV", "Revelation", "Rev", "NT", 66, ("Re", "The Revelation", "Apocalypse")),
)

BOOKS_BY_CODE: dict[str, BookInfo] = {b.code: b for b in BSB_BOOKS}


def _normalize_book_key(raw: str) -> str:
    """Normalize a book string by stripping punctuation and collapsing whitespace."""
    cleaned = re.sub(r"[.\-_]+", " ", raw.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _build_lookup_map() -> dict[str, BookInfo]:
    lookup: dict[str, BookInfo] = {}
    for book in BSB_BOOKS:
        candidates = [book.code, book.name, book.short_abbr, *book.aliases]
        for candidate in candidates:
            norm = _normalize_book_key(candidate)
            lookup[norm] = book
            lookup[norm.replace(" ", "")] = book
    return lookup


_BOOK_LOOKUP_MAP: dict[str, BookInfo] = _build_lookup_map()


def get_book_by_code(code: str) -> BookInfo:
    """Return canonical BookInfo by 3-letter USJ code (e.g. 'GEN', '1CO')."""
    upper = code.strip().upper()
    if upper not in BOOKS_BY_CODE:
        raise KeyError(f"Unknown BSB book code: {code!r}")
    return BOOKS_BY_CODE[upper]


def resolve_book(name_or_abbr: str) -> BookInfo | None:
    """Resolve any book name, USJ code, or common abbreviation to BookInfo."""
    if not name_or_abbr:
        return None
    norm = _normalize_book_key(name_or_abbr)
    if norm in _BOOK_LOOKUP_MAP:
        return _BOOK_LOOKUP_MAP[norm]
    no_space = norm.replace(" ", "")
    return _BOOK_LOOKUP_MAP.get(no_space)


def format_canonical_ref(book_code: str, chapter: int, verse: int | None = None) -> str:
    """Format a canonical reference such as 'Genesis 1:1' or 'Psalm 23:1'."""
    book = get_book_by_code(book_code)
    display_name = "Psalm" if book.code == "PSA" else book.name
    if verse is None:
        return f"{display_name} {chapter}"
    return f"{display_name} {chapter}:{verse}"
