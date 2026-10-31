"""Thin HTTP client for the backend. The frontend never imports backend code."""
import os
import time

import requests

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def health() -> dict:
    return requests.get(f"{BASE_URL}/health", timeout=10).json()


def documents() -> list[dict]:
    return requests.get(f"{BASE_URL}/documents", timeout=30).json()


def chat(question: str, filename: str | None = None) -> dict:
    r = requests.post(f"{BASE_URL}/chat", json={"question": question, "filename": filename}, timeout=120)
    r.raise_for_status()
    return r.json()


def ingest_and_wait(name: str, data: bytes, force: bool = False, poll_s: float = 2.0) -> dict:
    """Upload a PDF, then block (polling) until the backend finishes. Returns the final job state."""
    r = requests.post(f"{BASE_URL}/ingest", params={"force": force},
                      files={"file": (name, data, "application/pdf")}, timeout=600)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    while True:
        job = requests.get(f"{BASE_URL}/ingest/{job_id}", timeout=30).json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(poll_s)
