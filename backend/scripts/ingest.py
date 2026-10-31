"""Usage: python scripts/ingest.py [folder_or_pdf ...] [--force]"""
import sys
from pathlib import Path

from ragbot.core import config
from ragbot.services.rag_service import RAGService


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    targets = [Path(a) for a in args] or [config.PDF_DIR]
    pdfs = []
    for t in targets:
        pdfs += sorted(t.glob("*.pdf")) if t.is_dir() else [t]
    if not pdfs:
        sys.exit(f"No PDFs found in {targets}")

    svc = RAGService()
    for pdf in pdfs:
        for ev in svc.ingest(pdf, force=force):
            print(f"[{pdf.name}] {ev['stage']}: {ev.get('msg', '')}")
    print("\nCorpus:")
    for d in svc.docs():
        print(f"  {d['filename']}: {d['pages']} pages, {d['chunks']} chunks")


if __name__ == "__main__":
    main()
