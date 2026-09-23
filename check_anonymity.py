#!/usr/bin/env python3
"""Audit an anonymous release for common identity and privacy leaks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
PRIVATE_TERMS_FILE = ROOT / '.private_terms.txt'

# These trees exist only in the working copy and must never enter the release.
LOCAL_ONLY_PREFIXES = (
    'basicsr/OSEDiff/',
    'basicsr/SeeSR/',
    'dinov3/',
    'datasets/',
    'pretrained_models/',
    'experiments/',
    'results/',
    'tb_logger/',
    'wandb/',
    'tmp/',
    'lmdb_preview/',
    'dmd_log/',
    'outputs/',
    'logs/',
)
LOCAL_ONLY_EXCEPTIONS = {
    'datasets/README.md',
    'pretrained_models/README.md',
}

TEXT_SUFFIXES = {
    '',
    '.bat',
    '.c',
    '.cc',
    '.cff',
    '.cfg',
    '.cmd',
    '.cpp',
    '.cu',
    '.cuh',
    '.h',
    '.hpp',
    '.ini',
    '.json',
    '.m',
    '.md',
    '.py',
    '.pyi',
    '.ps1',
    '.rst',
    '.sh',
    '.tex',
    '.toml',
    '.txt',
    '.xml',
    '.yaml',
    '.yml',
}

BUILTIN_PATTERNS = (
    (
        'email address',
        re.compile(
            r'(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+'
            r'@[A-Z0-9.-]+\.[A-Z]{2,}',
            re.IGNORECASE,
        ),
    ),
    (
        'Windows absolute path',
        re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]'),
    ),
    (
        'local POSIX absolute path',
        re.compile(
            r'(?<![A-Za-z0-9_])/'
            r'(?:home|Users|data|mnt|root|efs_mount|cache|scratch|workspace)/'
        ),
    ),
    (
        'private key',
        re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    ),
    (
        'AWS access key',
        re.compile(r'(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])'),
    ),
    (
        'GitHub access token',
        re.compile(r'(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{20,}'),
    ),
    (
        'OpenAI-style secret key',
        re.compile(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}'),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    excerpt: str


def _run_git(
        arguments: list[str],
        *,
        check: bool = True,
        text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', '-C', str(ROOT), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        encoding='utf-8' if text else None,
        errors='replace' if text else None,
    )


def _is_repository() -> bool:
    result = _run_git(
        ['rev-parse', '--show-toplevel'], check=False, text=True)
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == ROOT.resolve()
    except OSError:
        return False


def _split_null_paths(payload: bytes) -> list[str]:
    return [
        value.decode('utf-8', errors='surrogateescape').replace('\\', '/')
        for value in payload.split(b'\0')
        if value
    ]


def _is_local_only(relative_path: str) -> bool:
    relative_path = relative_path.replace('\\', '/')
    if relative_path in LOCAL_ONLY_EXCEPTIONS:
        return False
    return any(relative_path.startswith(prefix)
               for prefix in LOCAL_ONLY_PREFIXES)


def _filesystem_candidates() -> list[str]:
    candidates = []
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith('.git/') or '__pycache__/' in relative:
            continue
        if _is_local_only(relative):
            continue
        candidates.append(relative)
    return sorted(candidates)


def _git_candidates(staged: bool) -> list[str]:
    if staged:
        result = _run_git(
            ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'])
    else:
        result = _run_git(
            ['ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    return sorted(set(_split_null_paths(result.stdout)))


def _read_candidate(relative_path: str, staged: bool) -> Optional[bytes]:
    if staged:
        result = _run_git(
            ['show', f':{relative_path}'], check=False)
        if result.returncode != 0:
            return None
        return result.stdout
    try:
        return (ROOT / relative_path).read_bytes()
    except OSError:
        return None


def _is_text_path(relative_path: str, payload: bytes) -> bool:
    path = Path(relative_path)
    if path.name in {'.gitignore', '.gitattributes'}:
        return True
    if path.name.startswith('LICENSE'):
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    return b'\0' not in payload[:8192]


def _private_terms() -> list[str]:
    if not PRIVATE_TERMS_FILE.is_file():
        return []
    terms = []
    for raw_line in PRIVATE_TERMS_FILE.read_text(
            encoding='utf-8', errors='replace').splitlines():
        term = raw_line.strip()
        if term and not term.startswith('#'):
            terms.append(term)
    return terms


def _safe_excerpt(line: str, start: int, end: int) -> str:
    excerpt = line[max(0, start - 30):min(len(line), end + 30)].strip()
    return excerpt.replace('\t', ' ')[:160]


def _scan_text(
        relative_path: str,
        text: str,
        private_terms: Iterable[str],
) -> list[Finding]:
    findings = []
    private_patterns = [
        (f'private term #{index}', re.compile(re.escape(term), re.IGNORECASE))
        for index, term in enumerate(private_terms, start=1)
    ]
    for line_number, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in (*BUILTIN_PATTERNS, *private_patterns):
            match = pattern.search(line)
            if match:
                findings.append(Finding(
                    relative_path,
                    line_number,
                    kind,
                    _safe_excerpt(line, match.start(), match.end()),
                ))
    return findings


def _audit_git_metadata() -> list[Finding]:
    if not _is_repository():
        return []

    findings = []
    name_result = _run_git(
        ['config', '--local', '--get', 'user.name'], check=False, text=True)
    email_result = _run_git(
        ['config', '--local', '--get', 'user.email'], check=False, text=True)
    local_name = name_result.stdout.strip()
    local_email = email_result.stdout.strip()
    if not local_name or local_name.casefold() not in {
            'anonymous', 'anonymous author'}:
        findings.append(Finding(
            '.git/config', 0, 'non-anonymous local Git author',
            local_name or '<not configured locally>',
        ))
    allowed_email = (
        local_email.casefold().endswith('@users.noreply.github.com')
        or local_email.casefold().endswith('.invalid')
    )
    if not local_email or not allowed_email:
        findings.append(Finding(
            '.git/config', 0, 'non-anonymous local Git email',
            local_email or '<not configured locally>',
        ))

    history = _run_git(
        ['log', '--all', '--format=%an%x09%ae'],
        check=False,
        text=True,
    )
    for entry in sorted(set(history.stdout.splitlines())):
        if not entry:
            continue
        name, _, email = entry.partition('\t')
        anonymous_name = name.casefold() in {'anonymous', 'anonymous author'}
        anonymous_email = (
            email.casefold().endswith('@users.noreply.github.com')
            or email.casefold().endswith('.invalid')
        )
        if not anonymous_name or not anonymous_email:
            findings.append(Finding(
                '<git history>', 0, 'identity-bearing commit author', entry))
    return findings


def audit(staged: bool) -> tuple[list[Finding], int]:
    in_repository = _is_repository()
    if staged and not in_repository:
        raise RuntimeError('--staged requires a Git repository.')

    candidates = (
        _git_candidates(staged) if in_repository
        else _filesystem_candidates())
    findings = []
    private_terms = _private_terms()
    scanned = 0

    for relative_path in candidates:
        findings.extend(_scan_text(
            f'<path:{relative_path}>', relative_path, private_terms))
        if _is_local_only(relative_path):
            findings.append(Finding(
                relative_path,
                0,
                'local-only path included in release',
                relative_path,
            ))
            continue
        payload = _read_candidate(relative_path, staged)
        if payload is None or not _is_text_path(relative_path, payload):
            continue
        scanned += 1
        text = payload.decode('utf-8-sig', errors='replace')
        findings.extend(_scan_text(relative_path, text, private_terms))

    findings.extend(_audit_git_metadata())
    return findings, scanned


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Check an anonymous release for common privacy leaks.')
    parser.add_argument(
        '--staged',
        action='store_true',
        help='scan staged Git contents instead of the working tree',
    )
    args = parser.parse_args()

    try:
        findings, scanned = audit(args.staged)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Anonymity audit failed to run: {error}', file=sys.stderr)
        return 2

    if findings:
        print(f'Anonymity audit found {len(findings)} issue(s):')
        for finding in findings:
            location = (
                f'{finding.path}:{finding.line}'
                if finding.line else finding.path)
            print(f'  {location}: {finding.kind}: {finding.excerpt}')
        return 1

    scope = 'staged' if args.staged else 'candidate'
    print(
        f'Anonymity audit passed: {scanned} {scope} text files scanned; '
        'no common privacy signatures found.')
    if not PRIVATE_TERMS_FILE.is_file():
        print(
            'Tip: add private names and account handles to the ignored '
            '.private_terms.txt for a project-specific check.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
