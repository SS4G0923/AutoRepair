"""Dependency-free PDF writer for exporting repair reports.

This intentionally avoids pulling a new heavy dependency (reportlab, weasyprint)
because the surrounding project already vendors-in minimal helpers and the
report layout is simple: a title, metadata table, and monospace code / diff
blocks. The generated PDF is rendered with built-in Helvetica / Courier fonts.

Format reference: PDF 1.4, minimal stream-compressed single-file output.
"""

from __future__ import annotations

import hashlib
import io
import textwrap
import zlib
from dataclasses import dataclass, field
from typing import Iterable


_PAGE_WIDTH = 612   # 8.5" x 72 dpi
_PAGE_HEIGHT = 792  # 11"  x 72 dpi
_MARGIN_X = 54      # 0.75"
_MARGIN_TOP = 752
_MARGIN_BOTTOM = 54
_DEFAULT_BODY_FONT = "F1"   # Helvetica
_DEFAULT_MONO_FONT = "F2"   # Courier
_DEFAULT_BOLD_FONT = "F3"   # Helvetica-Bold


@dataclass
class PdfLine:
    text: str
    font: str = _DEFAULT_BODY_FONT
    size: int = 11
    indent: int = 0
    is_heading: bool = False


@dataclass
class PdfDocument:
    title: str
    lines: list[PdfLine] = field(default_factory=list)

    def add_heading(self, text: str, *, size: int = 16) -> None:
        self.lines.append(PdfLine(text=text, font=_DEFAULT_BOLD_FONT, size=size, is_heading=True))
        self.lines.append(PdfLine(text="", size=4))

    def add_subheading(self, text: str) -> None:
        self.lines.append(PdfLine(text=text, font=_DEFAULT_BOLD_FONT, size=13, is_heading=True))
        self.lines.append(PdfLine(text="", size=2))

    def add_paragraph(self, text: str, *, indent: int = 0) -> None:
        wrapped = textwrap.wrap(text, width=88) if text else [""]
        for line in wrapped:
            self.lines.append(PdfLine(text=line, indent=indent))

    def add_meta(self, label: str, value: str) -> None:
        self.lines.append(PdfLine(
            text=f"{label}: {value}",
            indent=0,
            font=_DEFAULT_BODY_FONT,
            size=11,
        ))

    def add_code_block(self, content: str, *, title: str | None = None) -> None:
        if title:
            self.add_subheading(title)
        safe = content.replace("\t", "    ")
        for raw_line in safe.splitlines() or [""]:
            for chunk in textwrap.wrap(raw_line, width=92, drop_whitespace=False, replace_whitespace=False) or [""]:
                self.lines.append(PdfLine(text=chunk, font=_DEFAULT_MONO_FONT, size=9, indent=0))
        self.lines.append(PdfLine(text="", size=4))


def _escape_pdf_text(text: str) -> str:
    """Escape a string for PDF string literals."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _ascii_safe(text: str) -> str:
    """PDF built-in fonts only support WinAnsiEncoding; fall back to `?` otherwise."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _build_content_stream(pages: list[list[PdfLine]]) -> list[bytes]:
    streams: list[bytes] = []
    for page_lines in pages:
        buffer = io.StringIO()
        buffer.write("BT\n")
        current_font = None
        current_size = None
        y = _MARGIN_TOP
        first_line = True
        for entry in page_lines:
            line_height = max(int(entry.size * 1.35), 11)
            if first_line:
                buffer.write(f"1 0 0 1 {_MARGIN_X + entry.indent} {y} Tm\n")
                first_line = False
            else:
                buffer.write(f"1 0 0 1 {_MARGIN_X + entry.indent} {y} Tm\n")
            font_changed = (entry.font != current_font) or (entry.size != current_size)
            if font_changed:
                buffer.write(f"/{entry.font} {entry.size} Tf\n")
                current_font = entry.font
                current_size = entry.size
            text = _escape_pdf_text(_ascii_safe(entry.text))
            buffer.write(f"({text}) Tj\n")
            y -= line_height
        buffer.write("ET\n")
        streams.append(buffer.getvalue().encode("latin-1"))
    return streams


def _paginate(lines: list[PdfLine]) -> list[list[PdfLine]]:
    pages: list[list[PdfLine]] = []
    current: list[PdfLine] = []
    y = _MARGIN_TOP
    for entry in lines:
        height = max(int(entry.size * 1.35), 11)
        if y - height < _MARGIN_BOTTOM:
            pages.append(current)
            current = []
            y = _MARGIN_TOP
        current.append(entry)
        y -= height
    if current:
        pages.append(current)
    if not pages:
        pages.append([PdfLine(text="(empty document)")])
    return pages


def build_pdf_bytes(document: PdfDocument) -> bytes:
    pages = _paginate(document.lines)
    content_streams = _build_content_stream(pages)

    compressed_streams = [zlib.compress(stream) for stream in content_streams]

    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_f1 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_f2 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    font_f3 = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

    font_dict = (
        f"<< /F1 {font_f1} 0 R /F2 {font_f2} 0 R /F3 {font_f3} 0 R >>"
    ).encode("latin-1")
    resources = f"<< /Font {font_dict.decode('latin-1')} >>".encode("latin-1")

    page_ids: list[int] = []
    content_ids: list[int] = []

    pages_root_id = len(objects) + 1 + len(compressed_streams) * 2

    for stream in compressed_streams:
        stream_obj = (
            b"<< /Length " + str(len(stream)).encode("latin-1") +
            b" /Filter /FlateDecode >>\nstream\n" + stream + b"\nendstream"
        )
        stream_id = add_object(stream_obj)
        content_ids.append(stream_id)
        page_obj = (
            f"<< /Type /Page /Parent {pages_root_id} 0 R "
            f"/MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
            f"/Resources {resources.decode('latin-1')} "
            f"/Contents {stream_id} 0 R >>"
        ).encode("latin-1")
        page_id = add_object(page_obj)
        page_ids.append(page_id)

    kids_ref = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = (
        f"<< /Type /Pages /Kids [{kids_ref}] /Count {len(page_ids)} >>"
    ).encode("latin-1")
    pages_id = add_object(pages_obj)
    assert pages_id == pages_root_id, "Pages root id mismatch"

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf_buffer = io.BytesIO()
    pdf_buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: list[int] = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(pdf_buffer.tell())
        pdf_buffer.write(f"{index} 0 obj\n".encode("latin-1"))
        pdf_buffer.write(body)
        pdf_buffer.write(b"\nendobj\n")

    xref_offset = pdf_buffer.tell()
    pdf_buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf_buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_buffer.write(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf_buffer.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
        .encode("latin-1")
    )
    return pdf_buffer.getvalue()


def document_sha256(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def chain_paragraphs(lines: Iterable[str]) -> list[str]:
    return [line for line in lines if line is not None]
