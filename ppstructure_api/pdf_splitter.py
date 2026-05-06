from __future__ import annotations

import os
from pathlib import Path

from pdf2image import convert_from_path
from PIL import ImageOps


class PDFPageSplitter:
    def __init__(self, pdf_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]):
        self.pdf_path = Path(pdf_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()

    def split(self) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pages = convert_from_path(
            str(self.pdf_path),
            dpi=300,
            use_cropbox=False,
            strict=False,
        )

        saved_paths: list[Path] = []
        pdf_name = self.pdf_path.stem
        for index, page in enumerate(pages):
            page = ImageOps.exif_transpose(page)
            output_path = self.output_dir / f"{pdf_name}_page{index + 1}.png"
            page.save(output_path, "PNG")
            saved_paths.append(output_path)

        return saved_paths
