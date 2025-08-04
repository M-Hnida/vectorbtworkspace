#!/usr/bin/env python3
"""
CLI pour exploiter l'API RAG Assistant intégralement en ligne de commande.

Fonctionnalités:
- Sous-commandes: vectorbt, review-code, review-file
- Options globales: --base-url, --timeout, --retries, --retry-wait, --output, --headers
- Lecture du code via argument, fichier ou STDIN (piping)
- Timeouts, retries, gestion d'erreurs et codes de sortie
- Sortie JSON ou pretty

Exemples:
  Poser une question VectorBT:
    python vectoragent.py vectorbt -q "How to plot drawdown?" --base-url http://localhost:8000

  Review d’un snippet via STDIN:
    type mycode.py | python vectoragent.py review-code -q "optimize me"

  Review d’un fichier:
    python vectoragent.py review-file -f src/main.py -q "Any bugs?" --output json

  Retries et en-têtes:
    python vectoragent.py vectorbt -q "..." --retries 3 --retry-wait 2 --headers "{\"Authorization\":\"Bearer ABC\"}"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NETWORK = 2
EXIT_HTTP = 3
EXIT_FILE = 4
EXIT_PARSE = 5


def parse_headers(headers_json: str | None) -> dict[str, str]:
    if not headers_json:
        return {}
    try:
        parsed = json.loads(headers_json)
        if not isinstance(parsed, dict):
            raise ValueError("headers must be a JSON object")
        # Convert all values to string for requests headers
        return {str(k): str(v) for k, v in parsed.items()}
    except Exception as e:
        print(f"[ERROR] Invalid --headers JSON: {e}", file=sys.stderr)
        sys.exit(EXIT_PARSE)


def print_output(data: Any, output: str) -> None:
    if output == "json":
        print(json.dumps(data, ensure_ascii=False))
    else:
        # pretty
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(str(data))


def request_with_retries(method: str,
                         url: str,
                         *,
                         json_payload: dict | None = None,
                         headers: dict | None = None,
                         timeout: float = 15.0,
                         retries: int = 0,
                         retry_wait: float = 1.0) -> requests.Response:
    attempt = 0
    last_exc: Exception | None = None
    while True:
        try:
            resp = requests.request(method=method, url=url,
                                    json=json_payload, headers=headers, timeout=timeout)
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt >= retries:
                raise last_exc
            attempt += 1
            time.sleep(retry_wait)


def do_vectorbt(base_url: str, question: str, *, timeout: float, retries: int, retry_wait: float,
                headers: dict[str, str], output: str) -> int:
    url = f"{base_url.rstrip('/')}/query/vectorbt"
    payload = {"question": question}
    try:
        resp = request_with_retries("POST", url,
                                    json_payload=payload,
                                    headers=headers,
                                    timeout=timeout,
                                    retries=retries,
                                    retry_wait=retry_wait)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"[ERROR] Réseau/timeout: {e}", file=sys.stderr)
        return EXIT_NETWORK
    except Exception as e:
        print(f"[ERROR] Erreur inattendue réseau: {e}", file=sys.stderr)
        return EXIT_NETWORK

    if not resp.ok:
        # Tente d'afficher JSON d'erreur si présent
        try:
            err = resp.json()
        except Exception:
            err = {"status_code": resp.status_code, "text": resp.text}
        print_output({"error": "HTTP error", "details": err}, output)
        return EXIT_HTTP

    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    print_output(data, output)
    return EXIT_OK


def do_review_code(base_url: str, code: str, question: str, *, timeout: float, retries: int,
                   retry_wait: float, headers: dict[str, str], output: str) -> int:
    url = f"{base_url.rstrip('/')}/review/code"
    payload = {"code": code, "question": question}

    try:
        resp = request_with_retries("POST", url,
                                    json_payload=payload,
                                    headers=headers,
                                    timeout=timeout,
                                    retries=retries,
                                    retry_wait=retry_wait)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"[ERROR] Réseau/timeout: {e}", file=sys.stderr)
        return EXIT_NETWORK
    except Exception as e:
        print(f"[ERROR] Erreur inattendue réseau: {e}", file=sys.stderr)
        return EXIT_NETWORK

    if not resp.ok:
        try:
            err = resp.json()
        except Exception:
            err = {"status_code": resp.status_code, "text": resp.text}
        print_output({"error": "HTTP error", "details": err}, output)
        return EXIT_HTTP

    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    print_output(data, output)
    return EXIT_OK


def do_review_file(base_url: str, file_path: str, question: str, *, timeout: float, retries: int,
                   retry_wait: float, headers: dict[str, str], output: str) -> int:
    try:
        with open(file_path, encoding="utf-8") as fh:
            code = fh.read()
    except FileNotFoundError:
        print(f"[ERROR] Fichier introuvable: {file_path}", file=sys.stderr)
        return EXIT_FILE
    except Exception as e:
        print(f"[ERROR] Erreur lecture fichier {file_path}: {e}", file=sys.stderr)
        return EXIT_FILE

    return do_review_code(base_url, code, question,
                          timeout=timeout, retries=retries, retry_wait=retry_wait,
                          headers=headers, output=output)


def read_stdin_if_available() -> str | None:
    # Si on pipe des données, stdin n'est pas un TTY
    if not sys.stdin.isatty():
        try:
            return sys.stdin.read()
        except Exception:
            return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI pour interagir avec l'API RAG Assistant."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"URL de base de l'API (défaut: {DEFAULT_BASE_URL})")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Timeout requête en secondes (défaut: 15)")
    parser.add_argument("--retries", type=int, default=0,
                        help="Nombre de tentatives supplémentaires en cas d'échec réseau (défaut: 0)")
    parser.add_argument("--retry-wait", type=float, default=1.0,
                        help="Attente en secondes entre les retries (défaut: 1)")
    parser.add_argument("--output", choices=["pretty", "json"], default="pretty",
                        help="Format de sortie (pretty|json) (défaut: pretty)")
    parser.add_argument("--headers", type=str, default=None,
                        help='En-têtes HTTP au format JSON, ex: {"Authorization":"Bearer TOKEN"}')

    sub = parser.add_subparsers(dest="command", required=True)

    # vectorbt
    p_vec = sub.add_parser("vectorbt", help="Poser une question à l'assistant VectorBT")
    p_vec.add_argument("-q", "--question", required=True, help="Question à poser")

    # review-code
    p_rc = sub.add_parser("review-code", help="Soumettre du code (argument/STDIN) + question")
    p_rc.add_argument("-q", "--question", required=True, help="Question pour la relecture")
    p_rc.add_argument("-c", "--code", required=False,
                      help="Code à relire (si non fourni, tentative via STDIN)")

    # review-file
    p_rf = sub.add_parser("review-file", help="Relire un fichier + question")
    p_rf.add_argument("-f", "--file", required=True, help="Chemin du fichier à relire")
    p_rf.add_argument("-q", "--question", required=True, help="Question pour la relecture")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    base_url: str = args.base_url
    timeout: float = args.timeout
    retries: int = args.retries
    retry_wait: float = args.retry_wait
    output: str = args.output
    headers: dict[str, str] = parse_headers(args.headers)

    if args.command == "vectorbt":
        return do_vectorbt(base_url, args.question,
                           timeout=timeout, retries=retries, retry_wait=retry_wait,
                           headers=headers, output=output)

    if args.command == "review-code":
        code: str | None = args.code
        if not code:
            code = read_stdin_if_available()
        if not code:
            print("[ERROR] Aucun code fourni. Utilise --code ou pipe STDIN.", file=sys.stderr)
            return EXIT_USAGE
        return do_review_code(base_url, code, args.question,
                              timeout=timeout, retries=retries, retry_wait=retry_wait,
                              headers=headers, output=output)

    if args.command == "review-file":
        return do_review_file(base_url, args.file, args.question,
                              timeout=timeout, retries=retries, retry_wait=retry_wait,
                              headers=headers, output=output)

    # Ne devrait pas arriver grâce à required=True
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
