"""Extract text from PDF files using only the Python standard library.

This is a compact, best-effort reader so that longdoc can load ordinary text PDFs with no
third-party package. It handles:

* classic and compressed (xref stream / object stream) files, found by scanning, so a damaged
  cross-reference table does not matter;
* ``FlateDecode``, ``ASCII85Decode`` and ``ASCIIHexDecode`` streams, with PNG predictors;
* simple fonts (WinAnsi, MacRoman, ``/Differences``) and composite fonts through ``ToUnicode``
  CMaps, which covers most generated and word-processor PDFs;
* text positioning, so lines and words come out in reading order.

It does **not** handle: encrypted files, scanned pages (there is no text to read; OCR is needed),
fonts with neither an encoding nor a ``ToUnicode`` map, Type 3 fonts, or multi-column reading
order. For difficult files install ``pypdf`` (``pip install pypdf``): longdoc then uses it
automatically.
"""

from __future__ import annotations

# ruff: noqa: RUF001  (the glyph table holds real typographic characters on purpose)
import base64
import binascii
import re
import unicodedata
import zlib
from collections.abc import Iterator
from typing import Any

from .exceptions import DocumentLoadError, UnsupportedFormatError

_NAME_END = rb"\x00\t\n\f\r ()<>\[\]{}/%"
_TOKEN = re.compile(
    rb"(?P<ws>[\x00\t\n\f\r ]+|%[^\r\n]*)"
    rb"|(?P<name>/[^" + _NAME_END + rb"]*)"
    rb"|(?P<num>[+-]?(?:\d+\.?\d*|\.\d+))(?![^" + _NAME_END + rb"])"
    rb"|(?P<dstart><<)|(?P<dend>>>)"
    rb"|(?P<hex><[0-9A-Fa-f\s]*>)"
    rb"|(?P<lstr>\()"
    rb"|(?P<brk>[\[\]])"
    rb"|(?P<kw>[^" + _NAME_END + rb"]+)"
)
_OBJ = re.compile(rb"(?<![\d.])(\d+)[ \t\r\n]+(\d+)[ \t\r\n]+obj\b")
_ESCAPES = {
    ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t", ord("b"): b"\b", ord("f"): b"\f",
    ord("("): b"(", ord(")"): b")", ord("\\"): b"\\",
}  # fmt: skip
_MAX_DEPTH = 64


class _Name(str):
    """A PDF name object such as ``/Type``."""


class _Op(str):
    """A content-stream operator such as ``Tj``."""


class _Ref(tuple):  # type: ignore[type-arg]
    """An indirect reference ``n g R``."""


_END_ARRAY = object()
_END_DICT = object()
_EOF = object()


class _Lexer:
    """Turns PDF syntax into Python values (dict, list, bytes, int, float, _Name, _Op)."""

    def __init__(self, data: bytes, pos: int = 0, *, refs: bool = True) -> None:
        self.data = data
        self.pos = pos
        self.refs = refs

    def token(self) -> Any:
        data, n = self.data, len(self.data)
        while self.pos < n:
            m = _TOKEN.match(data, self.pos)
            if m is None:
                self.pos += 1
                continue
            self.pos = m.end()
            kind = m.lastgroup
            if kind == "ws":
                continue
            if kind == "name":
                return _Name(_unescape_name(m.group()[1:]))
            if kind == "num":
                text = m.group()
                return float(text) if b"." in text else int(text)
            if kind == "dstart":
                return self._dict()
            if kind == "dend":
                return _END_DICT
            if kind == "hex":
                return _hex_bytes(m.group()[1:-1])
            if kind == "lstr":
                return self._literal()
            if kind == "brk":
                return self._array() if m.group() == b"[" else _END_ARRAY
            word = m.group()
            if word == b"true":
                return True
            if word == b"false":
                return False
            if word == b"null":
                return None
            return _Op(word.decode("latin-1"))
        return _EOF

    def value(self, depth: int = 0) -> Any:
        """Read one value, resolving ``n g R`` into a :class:`_Ref` when enabled."""
        if depth > _MAX_DEPTH:
            raise DocumentLoadError("PDF structure is nested too deeply.")
        tok = self.token()
        if self.refs and isinstance(tok, int) and not isinstance(tok, bool):
            save = self.pos
            second = self.token()
            if isinstance(second, int) and not isinstance(second, bool):
                third = self.token()
                if isinstance(third, _Op) and third == "R":
                    return _Ref((tok, second))
            self.pos = save
        return tok

    def _array(self) -> list[Any]:
        items: list[Any] = []
        while True:
            tok = self.value()
            if tok is _END_ARRAY or tok is _EOF:
                return items
            items.append(tok)

    def _dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while True:
            key = self.token()
            if key is _END_DICT or key is _EOF:
                return out
            if not isinstance(key, _Name):
                continue
            val = self.value()
            if val is _END_DICT or val is _EOF:
                return out
            out[str(key)] = val

    def _literal(self) -> bytes:
        data, pos, depth = self.data, self.pos, 1
        out = bytearray()
        n = len(data)
        while pos < n:
            c = data[pos]
            pos += 1
            if c == 0x5C:  # backslash
                if pos >= n:
                    break
                e = data[pos]
                pos += 1
                if e in _ESCAPES:
                    out += _ESCAPES[e]
                elif 0x30 <= e <= 0x37:  # octal
                    digits = bytes([e])
                    while len(digits) < 3 and pos < n and 0x30 <= data[pos] <= 0x37:
                        digits += bytes([data[pos]])
                        pos += 1
                    out.append(int(digits, 8) & 0xFF)
                elif e == 0x0D:  # line continuation
                    if pos < n and data[pos] == 0x0A:
                        pos += 1
                elif e != 0x0A:
                    out.append(e)
            elif c == 0x28:
                depth += 1
                out.append(c)
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    break
                out.append(c)
            else:
                out.append(c)
        self.pos = pos
        return bytes(out)


def _unescape_name(raw: bytes) -> str:
    return re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw).decode(
        "latin-1"
    )


def _hex_bytes(raw: bytes) -> bytes:
    digits = re.sub(rb"\s+", b"", raw)
    if len(digits) % 2:
        digits += b"0"
    return binascii.unhexlify(digits)


# -- filters --------------------------------------------------------------------------------


def _predictor(data: bytes, params: dict[str, Any]) -> bytes:
    predictor = int(params.get("Predictor", 1))
    if predictor < 10:
        return data
    columns = int(params.get("Columns", 1))
    colors = int(params.get("Colors", 1))
    bits = int(params.get("BitsPerComponent", 8))
    bpp = max(1, colors * bits // 8)
    row_len = (columns * colors * bits + 7) // 8
    out = bytearray()
    prev = bytearray(row_len)
    step = row_len + 1
    for start in range(0, len(data) - step + 1, step):
        ftype = data[start]
        row = bytearray(data[start + 1 : start + step])
        for i in range(row_len):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            upleft = prev[i - bpp] if i >= bpp else 0
            if ftype == 1:
                row[i] = (row[i] + left) & 0xFF
            elif ftype == 2:
                row[i] = (row[i] + up) & 0xFF
            elif ftype == 3:
                row[i] = (row[i] + (left + up) // 2) & 0xFF
            elif ftype == 4:
                p = left + up - upleft
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                pred = left if pa <= pb and pa <= pc else (up if pb <= pc else upleft)
                row[i] = (row[i] + pred) & 0xFF
        out += row
        prev = row
    return bytes(out)


def _apply_filters(raw: bytes, filters: Any, parms: Any) -> bytes | None:
    """Decode a stream. Returns ``None`` if it uses a filter this reader does not support."""
    names = filters if isinstance(filters, list) else ([filters] if filters else [])
    plist = parms if isinstance(parms, list) else [parms] * len(names)
    data = raw
    for name, parm in zip(names, plist, strict=False):
        parm = parm if isinstance(parm, dict) else {}
        name = str(name)
        if name in ("FlateDecode", "Fl"):
            try:
                data = zlib.decompressobj().decompress(data)
            except zlib.error:
                try:  # truncated or slightly corrupt streams: keep what decodes
                    data = zlib.decompressobj().decompress(data[2:], -zlib.MAX_WBITS)
                except zlib.error:
                    return None
            data = _predictor(data, parm)
        elif name in ("ASCIIHexDecode", "AHx"):
            data = _hex_bytes(data.split(b">")[0])
        elif name in ("ASCII85Decode", "A85"):
            body = data.strip()
            if body.startswith(b"<~"):
                body = body[2:]
            body = body.split(b"~>")[0]
            try:
                data = base64.a85decode(body)
            except ValueError:
                return None
        else:
            return None  # LZW, RunLength, image codecs, crypt: not supported
    return data


# -- fonts ----------------------------------------------------------------------------------

_GLYPHS = {
    "space": " ", "period": ".", "comma": ",", "colon": ":", "semicolon": ";", "hyphen": "-",
    "minus": "-", "underscore": "_", "slash": "/", "backslash": "\\", "parenleft": "(",
    "parenright": ")", "bracketleft": "[", "bracketright": "]", "braceleft": "{",
    "braceright": "}", "quotesingle": "'", "quotedbl": '"', "quoteright": "’",
    "quoteleft": "‘", "quotedblleft": "“", "quotedblright": "”",
    "exclam": "!", "question": "?", "at": "@", "numbersign": "#", "dollar": "$",
    "percent": "%", "ampersand": "&", "asterisk": "*", "plus": "+", "equal": "=",
    "less": "<", "greater": ">", "bar": "|", "asciitilde": "~", "asciicircum": "^",
    "grave": "`", "bullet": "•", "endash": "–", "emdash": "—",
    "ellipsis": "…", "fi": "fi", "fl": "fl", "ff": "ff", "ffi": "ffi", "ffl": "ffl",
    "section": "§", "paragraph": "¶", "copyright": "©", "registered": "®",
    "trademark": "™", "degree": "°", "Euro": "€", "sterling": "£",
    "yen": "¥", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip


_ACCENTS = {
    "acute": "ACUTE", "grave": "GRAVE", "circumflex": "CIRCUMFLEX", "dieresis": "DIAERESIS",
    "tilde": "TILDE", "ring": "RING ABOVE", "cedilla": "CEDILLA", "caron": "CARON",
    "ogonek": "OGONEK", "macron": "MACRON", "breve": "BREVE", "dotaccent": "DOT ABOVE",
    "hungarumlaut": "DOUBLE ACUTE", "stroke": "STROKE", "slash": "STROKE",
}  # fmt: skip
_GREEK = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
    "chi", "psi", "omega",
)  # fmt: skip
_MORE_GLYPHS = {
    "approxequal": "\u2248", "notequal": "\u2260", "lessequal": "\u2264", "greaterequal": "\u2265",
    "infinity": "\u221e", "plusminus": "\u00b1", "multiply": "\u00d7", "divide": "\u00f7",
    "periodcentered": "\u00b7", "guillemotleft": "\u00ab", "guillemotright": "\u00bb",
    "guilsinglleft": "\u2039", "guilsinglright": "\u203a", "cent": "\u00a2", "florin": "\u0192",
    "onehalf": "\u00bd", "onequarter": "\u00bc", "threequarters": "\u00be",
    "ordfeminine": "\u00aa", "ordmasculine": "\u00ba", "germandbls": "\u00df", "AE": "\u00c6",
    "ae": "\u00e6", "OE": "\u0152", "oe": "\u0153", "Oslash": "\u00d8", "oslash": "\u00f8",
    "Lslash": "\u0141", "lslash": "\u0142", "Scaron": "\u0160", "scaron": "\u0161",
    "Zcaron": "\u017d", "zcaron": "\u017e", "Ydieresis": "\u0178", "dagger": "\u2020",
    "daggerdbl": "\u2021", "perthousand": "\u2030", "quotesinglbase": "\u201a",
    "quotedblbase": "\u201e", "arrowright": "\u2192", "arrowleft": "\u2190", "checkmark": "\u2713",
    "nbspace": "\u00a0", "hyphenminus": "-", "sfthyphen": "\u00ad", "mu": "\u00b5",
    "logicalnot": "\u00ac", "brokenbar": "\u00a6", "dotlessi": "\u0131", "Eth": "\u00d0",
    "eth": "\u00f0", "Thorn": "\u00de", "thorn": "\u00fe", "currency": "\u00a4",
}  # fmt: skip


def _glyph_char(name: str) -> str:
    if name in _GLYPHS:
        return _GLYPHS[name]
    if name in _MORE_GLYPHS:
        return _MORE_GLYPHS[name]
    if len(name) == 1:
        return name
    m = re.fullmatch(r"([A-Za-z])(" + "|".join(_ACCENTS) + ")", name)
    if m:
        case = "CAPITAL" if m.group(1).isupper() else "SMALL"
        try:
            return unicodedata.lookup(
                f"LATIN {case} LETTER {m.group(1).upper()} WITH {_ACCENTS[m.group(2)]}"
            )
        except KeyError:
            pass
    if name.lower() in _GREEK:
        case = "CAPITAL" if name[0].isupper() else "SMALL"
        try:
            return unicodedata.lookup(f"GREEK {case} LETTER {name.upper()}")
        except KeyError:
            pass
    m = re.fullmatch(r"uni([0-9A-Fa-f]{4})", name)
    if m:
        return chr(int(m.group(1), 16))
    m = re.fullmatch(r"u([0-9A-Fa-f]{4,6})", name)
    if m:
        return chr(int(m.group(1), 16))
    return ""


def _symbol_table() -> dict[int, str]:
    """The built-in encoding of the standard Symbol font (Greek letters and math signs)."""
    table: dict[int, str] = {}
    upper = "ABGDEZHQIKLMNXOPRSTUFCYW"  # Symbol maps Latin keys onto Greek letters
    greek_upper = "\u0391\u0392\u0393\u0394\u0395\u0396\u0397\u0398\u0399\u039a\u039b\u039c"
    greek_upper += "\u039d\u039e\u039f\u03a0\u03a1\u03a3\u03a4\u03a5\u03a6\u03a7\u03a8\u03a9"
    greek_lower = greek_upper.lower().replace("\u03c2", "\u03c3")
    for key, up, low in zip(upper, greek_upper, greek_lower, strict=True):
        table[ord(key)] = up
        table[ord(key.lower())] = low
    table.update(
        {
            0x20: " ", 0x22: "\u2200", 0x24: "\u2203", 0x27: "\u220b", 0x2A: "\u2217",
            0x2D: "\u2212", 0x40: "\u2245", 0x5C: "\u2234", 0x5E: "\u22a5", 0xA3: "\u2264",
            0xA5: "\u221e", 0xAB: "\u2194", 0xAC: "\u2190", 0xAD: "\u2191", 0xAE: "\u2192",
            0xAF: "\u2193", 0xB0: "\u00b0", 0xB1: "\u00b1", 0xB2: "\u2033", 0xB3: "\u2265",
            0xB4: "\u00d7", 0xB5: "\u221d", 0xB6: "\u2202", 0xB7: "\u2022", 0xB8: "\u00f7",
            0xB9: "\u2260", 0xBA: "\u2261", 0xBB: "\u2248", 0xBC: "\u2026", 0xC5: "\u2295",
            0xC6: "\u2205", 0xC7: "\u2229", 0xC8: "\u222a", 0xCE: "\u2208", 0xD0: "\u2220",
            0xD1: "\u2207", 0xD5: "\u220f", 0xD6: "\u221a", 0xD7: "\u22c5", 0xD8: "\u00ac",
            0xD9: "\u2227", 0xDA: "\u2228", 0xE5: "\u2211", 0xF2: "\u222b",
        }
    )  # fmt: skip
    return table


_SYMBOL = _symbol_table()
_ZAPF = {
    0x6C: "\u25cf", 0x6D: "\u274d", 0x6E: "\u25a0", 0x6F: "\u274f", 0x70: "\u2750",
    0x71: "\u2751", 0x72: "\u2752", 0x75: "\u25c6", 0x33: "\u2713", 0x34: "\u2714",
    0x35: "\u2715", 0x36: "\u2716", 0x37: "\u2717", 0x38: "\u2718", 0x48: "\u2605",
    0x49: "\u2606", 0x4E: "\u2736", 0xA8: "\u2663", 0xA9: "\u2666", 0xAA: "\u2665",
    0xAB: "\u2660",
}  # fmt: skip


def _parse_cmap(data: bytes) -> tuple[dict[int, str], int]:
    """Parse a ToUnicode CMap into ``({code: text}, code_length_in_bytes)``."""
    text = data.decode("latin-1")
    code_len = 1
    for m in re.finditer(r"begincodespacerange(.*?)endcodespacerange", text, re.DOTALL):
        for lo in re.findall(r"<([0-9A-Fa-f]+)>", m.group(1))[::2]:
            code_len = max(code_len, len(lo) // 2)

    def to_text(hexstr: str) -> str:
        raw = binascii.unhexlify(hexstr if len(hexstr) % 2 == 0 else hexstr + "0")
        try:
            return raw.decode("utf-16-be")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    mapping: dict[int, str] = {}
    for m in re.finditer(r"beginbfchar(.*?)endbfchar", text, re.DOTALL):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", m.group(1)):
            mapping[int(src, 16)] = to_text(dst)
    for m in re.finditer(r"beginbfrange(.*?)endbfrange", text, re.DOTALL):
        body = m.group(1)
        pattern = r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]*>)"
        for lo, hi, dst in re.findall(pattern, body):
            start, end = int(lo, 16), int(hi, 16)
            if end - start > 0xFFFF:
                continue
            if dst.startswith("["):
                for i, item in enumerate(re.findall(r"<([0-9A-Fa-f]*)>", dst)):
                    if start + i <= end:
                        mapping[start + i] = to_text(item)
            else:
                base = to_text(dst[1:-1])
                for i in range(end - start + 1):
                    mapping[start + i] = (base[:-1] + chr(ord(base[-1]) + i)) if base else ""
    return mapping, code_len


class _TextState:
    """Mutable text state while walking one page's content stream."""

    def __init__(self) -> None:
        self.font: _Font | None = None
        self.size = 1.0
        self.leading = 0.0
        self.y: float | None = None
        self.gap = False


class _Font:
    """Decodes the bytes of a text string into Unicode for one font."""

    def __init__(self, obj: dict[str, Any], doc: _Document) -> None:
        self.cmap: dict[int, str] = {}
        self.code_len = 1
        self.table: dict[int, str] = {}
        self.composite = str(obj.get("Subtype")) == "Type0"
        tu = doc.stream(obj.get("ToUnicode"))
        if tu is not None:
            self.cmap, self.code_len = _parse_cmap(tu)
        if self.composite and not self.cmap:
            self.code_len = 2
        enc = doc.resolve(obj.get("Encoding"))
        base = enc if isinstance(enc, _Name) else None
        diffs: list[Any] = []
        if isinstance(enc, dict):
            b = doc.resolve(enc.get("BaseEncoding"))
            base = b if isinstance(b, _Name) else None
            diffs = doc.resolve(enc.get("Differences")) or []
        codec = {"MacRomanEncoding": "mac_roman"}.get(str(base), "cp1252")
        for code in range(256):
            self.table[code] = bytes([code]).decode(codec, errors="replace")
        # The standard Symbol and ZapfDingbats fonts carry their own built-in encodings.
        base_font = str(obj.get("BaseFont", "")).split("+")[-1]
        if enc is None and base_font == "Symbol":
            self.table.update(_SYMBOL)
        elif enc is None and base_font == "ZapfDingbats":
            self.table.update(_ZAPF)
        code = 0
        for item in diffs:
            item = doc.resolve(item)
            if isinstance(item, int):
                code = item
            elif isinstance(item, _Name):
                ch = _glyph_char(str(item))
                if ch:
                    self.table[code] = ch
                code += 1

    def decode(self, raw: bytes) -> str:
        out: list[str] = []
        step = self.code_len
        if self.composite or self.cmap:
            for i in range(0, len(raw) - step + 1, step):
                code = int.from_bytes(raw[i : i + step], "big")
                if code in self.cmap:
                    out.append(self.cmap[code])
                elif not self.composite:
                    out.append(self.table.get(code, ""))
            return "".join(out)
        return "".join(self.table.get(b, "") for b in raw)


# -- document -------------------------------------------------------------------------------


class _Document:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offsets: dict[int, int] = {}
        self.cache: dict[int, Any] = {}
        self.embedded: dict[int, Any] = {}
        self._scan()

    def _scan(self) -> None:
        for m in _OBJ.finditer(self.data):
            self.offsets[int(m.group(1))] = m.end()  # later definitions win (incremental updates)
        # Compressed objects live inside object streams.
        for num in list(self.offsets):
            obj = self._load(num)
            if isinstance(obj, dict) and str(obj.get("Type")) == "ObjStm":
                self._unpack_objstm(obj)

    def _unpack_objstm(self, obj: dict[str, Any]) -> None:
        raw = self.stream(obj)
        if raw is None:
            return
        count, first = (
            int(self.resolve(obj.get("N")) or 0),
            int(self.resolve(obj.get("First")) or 0),
        )
        header = _Lexer(raw, refs=False)
        pairs = []
        for _ in range(count):
            num, off = header.token(), header.token()
            if isinstance(num, int) and isinstance(off, int):
                pairs.append((num, off))
        for num, off in pairs:
            if num not in self.offsets:
                self.embedded[num] = _Lexer(raw, first + off).value()

    def _load(self, num: int) -> Any:
        if num in self.cache:
            return self.cache[num]
        self.cache[num] = None  # cycle guard
        pos = self.offsets.get(num)
        if pos is None:
            value = self.embedded.get(num)
        else:
            lexer = _Lexer(self.data, pos)
            value = lexer.value()
            if isinstance(value, dict):
                value = self._attach_stream(value, lexer)
        self.cache[num] = value
        return value

    def _attach_stream(self, obj: dict[str, Any], lexer: _Lexer) -> Any:
        save = lexer.pos
        tok = lexer.token()
        if not (isinstance(tok, _Op) and tok == "stream"):
            lexer.pos = save
            return obj
        start = lexer.pos
        if self.data[start : start + 2] == b"\r\n":
            start += 2
        elif self.data[start : start + 1] in (b"\n", b"\r"):
            start += 1
        length = self.resolve(obj.get("Length"))
        end = -1
        if isinstance(length, int) and length >= 0:
            probe = self.data[start + length : start + length + 12].lstrip()
            if probe.startswith(b"endstream"):
                end = start + length
        if end < 0:
            end = self.data.find(b"endstream", start)
            if end < 0:
                end = len(self.data)
            while end > start and self.data[end - 1 : end] in (b"\n", b"\r"):
                end -= 1
        obj = dict(obj)
        obj["__stream__"] = self.data[start:end]
        return obj

    def resolve(self, value: Any, depth: int = 0) -> Any:
        while isinstance(value, _Ref) and depth < _MAX_DEPTH:
            value = self._load(value[0])
            depth += 1
        return value

    def stream(self, value: Any) -> bytes | None:
        obj = self.resolve(value)
        if not isinstance(obj, dict) or "__stream__" not in obj:
            return None
        return _apply_filters(
            obj["__stream__"],
            self.resolve(obj.get("Filter")),
            self.resolve(obj.get("DecodeParms") or obj.get("DP")),
        )

    def objects(self) -> Iterator[tuple[int, Any]]:
        for num in sorted(set(self.offsets) | set(self.embedded)):
            yield num, self._load(num)

    # -- structure ---------------------------------------------------------------------------

    def pages(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Return ``[(page_dict, resources)]`` in reading order."""
        catalog = next(
            (
                o
                for _, o in self.objects()
                if isinstance(o, dict) and str(o.get("Type")) == "Catalog"
            ),
            None,
        )
        found: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen: set[int] = set()

        def walk(node: Any, inherited: dict[str, Any], depth: int = 0) -> None:
            ref = node[0] if isinstance(node, _Ref) else None
            if ref is not None:
                if ref in seen or depth > _MAX_DEPTH:
                    return
                seen.add(ref)
            obj = self.resolve(node)
            if not isinstance(obj, dict):
                return
            res = self.resolve(obj.get("Resources")) or inherited
            kids = self.resolve(obj.get("Kids"))
            if isinstance(kids, list):
                for kid in kids:
                    walk(kid, res if isinstance(res, dict) else inherited, depth + 1)
            else:
                found.append((obj, res if isinstance(res, dict) else {}))

        if catalog is not None:
            walk(catalog.get("Pages"), {})
        if not found:  # no usable page tree: take every /Page object in file order
            for _, obj in self.objects():
                if isinstance(obj, dict) and str(obj.get("Type")) == "Page":
                    res = self.resolve(obj.get("Resources"))
                    found.append((obj, res if isinstance(res, dict) else {}))
        return found

    def page_content(self, page: dict[str, Any]) -> bytes:
        contents = self.resolve(page.get("Contents"))
        parts = contents if isinstance(contents, list) else [page.get("Contents")]
        chunks = []
        for part in parts:
            data = self.stream(part)
            if data:
                chunks.append(data)
        return b"\n".join(chunks)


# -- text extraction ---------------------------------------------------------------------------


# A text run that follows another on the same line gets a space between them, except around
# hyphens and closing punctuation, which word processors often draw as separate runs.
_NO_SPACE_BEFORE = (
    ",",
    ".",
    ";",
    ":",
    ")",
    "]",
    "}",
    "!",
    "?",
    "%",
    "-",
    "\u2010",
    "\u2011",
    "\u00ad",
)
_NO_SPACE_AFTER = ("(", "[", "{", "-", "\u2010", "\u2011", "\u00ad", "/")


def _mul(m1: list[float], m2: list[float]) -> list[float]:
    a, b, c, d, e, f = m1
    g, h, i, j, k, l = m2  # noqa: E741
    return [
        a * g + b * i, a * h + b * j, c * g + d * i, c * h + d * j,
        e * g + f * i + k, e * h + f * j + l,
    ]  # fmt: skip


def _page_text(doc: _Document, page: dict[str, Any], resources: dict[str, Any]) -> str:
    content = doc.page_content(page)
    if not content:
        return ""
    fonts: dict[str, _Font] = {}
    font_res = doc.resolve(resources.get("Font")) or {}

    def font_for(name: str) -> _Font | None:
        if name not in fonts:
            obj = doc.resolve(font_res.get(name)) if isinstance(font_res, dict) else None
            if isinstance(obj, dict):
                fonts[name] = _Font(obj, doc)
        return fonts.get(name)

    lines: list[list[str]] = [[]]
    st = _TextState()
    tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    tlm = list(tm)

    def emit(text: str) -> None:
        if not text:
            return
        y = tm[5]
        scale = abs(tlm[3]) or 1.0
        tolerance = max(1.0, 0.3 * st.size * scale)
        if st.y is not None and abs(y - st.y) > tolerance:
            lines.append([])
        elif st.y is not None and lines[-1]:
            prev = lines[-1][-1]
            if (
                st.gap
                and not prev.endswith((" ", "\n", *_NO_SPACE_AFTER))
                and not text.startswith((" ", *_NO_SPACE_BEFORE))
            ):
                lines[-1].append(" ")
        st.y = y
        st.gap = False
        lines[-1].append(text)

    def show(raw: bytes) -> None:
        emit(st.font.decode(raw) if st.font else raw.decode("latin-1"))

    lexer = _Lexer(content, refs=False)
    stack: list[Any] = []
    while True:
        tok = lexer.token()
        if tok is _EOF:
            break
        if not isinstance(tok, _Op):
            stack.append(tok)
            if len(stack) > 4096:
                stack.clear()
            continue
        op, args = str(tok), stack
        stack = []
        nums = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        try:
            if op == "BT":
                tm, tlm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif op == "Tf" and len(args) >= 2 and isinstance(args[-2], _Name):
                st.font = font_for(str(args[-2]))
                st.size = float(args[-1]) if isinstance(args[-1], (int, float)) else 1.0
            elif op == "Tm" and len(nums) >= 6:
                tm, tlm = [float(n) for n in nums[-6:]], [float(n) for n in nums[-6:]]
                st.gap = True
            elif op in ("Td", "TD") and len(nums) >= 2:
                if op == "TD":
                    st.leading = -nums[-1]
                tlm = _mul([1.0, 0.0, 0.0, 1.0, nums[-2], nums[-1]], tlm)
                tm = list(tlm)
                st.gap = True
            elif op == "TL" and nums:
                st.leading = nums[-1]
            elif op == "T*":
                tlm = _mul([1.0, 0.0, 0.0, 1.0, 0.0, -st.leading], tlm)
                tm = list(tlm)
            elif op == "Tj" and args and isinstance(args[-1], bytes):
                show(args[-1])
            elif op in ("'", '"') and args and isinstance(args[-1], bytes):
                tlm = _mul([1.0, 0.0, 0.0, 1.0, 0.0, -st.leading], tlm)
                tm = list(tlm)
                show(args[-1])
            elif op == "TJ" and args and isinstance(args[-1], list):
                for item in args[-1]:
                    if isinstance(item, bytes):
                        show(item)
                    elif isinstance(item, (int, float)) and item < -180 and lines[-1]:
                        lines[-1].append(" ")
            elif op == "BI":  # inline image: skip its binary data
                end = re.search(rb"\sEI(?=[\s]|$)", content[lexer.pos :])
                lexer.pos = lexer.pos + end.end() if end else len(content)
        except (ValueError, IndexError, TypeError):
            continue
    text_lines = ["".join(parts).strip() for parts in lines]
    return "\n".join(line for line in text_lines if line)


def extract_pages(data: bytes) -> list[str]:
    """Return the text of each page of a PDF given as bytes.

    Raises:
        DocumentLoadError: Not a PDF, or its structure could not be read.
        UnsupportedFormatError: The file is encrypted.
    """
    if not data.lstrip()[:1024].startswith(b"%PDF") and b"%PDF-" not in data[:1024]:
        raise DocumentLoadError("Not a PDF file (no %PDF header).")
    if re.search(rb"/Encrypt\b", data):
        raise UnsupportedFormatError(
            "This PDF is encrypted. Decrypt it first (or install pypdf with its crypto support)."
        )
    try:
        doc = _Document(data)
        pages = doc.pages()
        return [_page_text(doc, page, resources) for page, resources in pages]
    except (DocumentLoadError, UnsupportedFormatError):
        raise
    except (RecursionError, ValueError, KeyError, IndexError, TypeError, zlib.error) as exc:
        raise DocumentLoadError(f"Could not read this PDF ({type(exc).__name__}).") from exc
