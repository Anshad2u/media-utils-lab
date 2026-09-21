#!/usr/bin/env python3
"""
The private transcript archive: one repository, one deploy key, one layout.

Imported by both process.py, which writes a transcript, and digest.py, which
reads a day of them. The path a transcript is stored at, the branch it goes to,
and the ssh options that carry the credential are decided here and nowhere else,
so the writer and the reader cannot drift apart - a drift that would show up as
an archive that fills correctly and compiles nothing.

Why a repository rather than object storage
-------------------------------------------
It was Cloudflare R2 first. A repository is easier to reason about: the storage
is inspectable, the history is the audit trail, and the credential is a deploy
key that is scoped to this one repository and cannot do anything else. What it
is not is free of cost - see the clone note below - so the layout is built
around keeping that cost flat.

Why the clone is cheap and stays cheap
--------------------------------------
`--depth 1 --filter=tree:0 --sparse` together transfer the tip commit and the
trees along the path being touched, and nothing else. A plain clone would
transfer the entire archive on every single recording, and the archive only ever
grows, so the cost of a clone would grow with it. This way the transfer is a few
tens of kilobytes whether the archive holds ten transcripts or a hundred
thousand. The sparse path must be set before writing into it, or `git add`
refuses a path outside the cone.

What is never logged
--------------------
Nothing from this module is safe to print: not the transcript, not the key, not
the repository name, not the commit. `describe()` is here rather than in either
caller because it is the single sanitiser both of them depend on, and two copies
of a sanitiser is one copy that is out of date.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

HOST = "github.com"
DIR = "transcripts"
BRANCH = "main"


def sanitise(text: str) -> str:
    """Strip anything url-shaped or opaque out of a line bound for the log.

    These logs are world-readable and error text routinely embeds URLs - a
    requests error carries the whole request URL, and a Telegram URL carries the
    bot token - so URL-shaped text and long opaque strings go first.
    """
    text = str(text).replace("\n", " ")
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b", "<token>", text)
    text = re.sub(r"\bgsk_[A-Za-z0-9]{20,}\b", "<key>", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{40,}\b", "<opaque>", text)
    return text.strip()


def describe(error: BaseException) -> str:
    """A log-safe summary of an exception."""
    return f"{type(error).__name__}: {sanitise(error)[:300]}"


def settings() -> tuple[str, str]:
    """(owner/name, deploy key) when the archive is configured, else ("", "").

    Deliberately all-or-nothing: half a configuration is a mistake, and a
    mistake that silently writes nothing looks exactly like a day with no
    recordings. Both values are checked for content, not just presence, so a
    secret set to a blank line is treated as unset.
    """
    repo = os.environ.get("ARCHIVE_REPO", "").strip().strip("/")
    key = os.environ.get("ARCHIVE_SSH_KEY", "")
    if repo.count("/") != 1 or not key.strip():
        return "", ""
    return repo, key


def git_env(root: Path, key: str) -> dict:
    """An environment whose only credential is the archive's deploy key.

    The key is written to a file rather than passed as a variable because ssh
    has no option to take a private key from the environment, and it is written
    under a temp directory that the job destroys. `BatchMode=yes` turns a
    missing or rejected key into an immediate failure instead of a prompt that
    would hang until the job timeout.

    Written as bytes with explicit LF endings, not text. On a machine whose
    default newline is CRLF, `write_text` would turn every line ending into
    CRLF, and ssh then rejects the key with "error in libcrypto: unsupported"
    - a failure that looks like a revoked key and is not.
    """
    identity = root / "deploy_key"
    text = key.replace("\r\n", "\n")
    if not text.endswith("\n"):
        # ssh will not read a key whose last line has no terminator, and the
        # error it gives for that is the same "unsupported" as a malformed one.
        text += "\n"
    identity.write_bytes(text.encode("utf-8"))
    os.chmod(identity, 0o600)
    return {
        **os.environ,
        "GIT_SSH_COMMAND": (
            f"ssh -i {shlex.quote(str(identity))} -o IdentitiesOnly=yes "
            f"-o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            f"-o UserKnownHostsFile={shlex.quote(str(root / 'known_hosts'))}"
        ),
        "GIT_TERMINAL_PROMPT": "0",
        # git would otherwise pick up a runner-wide credential helper and try to
        # authenticate as somebody else. The deploy key is the only credential.
        "GIT_ASKPASS": "",
    }


def git(log, env: dict, *args: str, cwd: Path | None = None, timeout: float = 180.0) -> bool:
    """Run one git command. Returns False rather than raising, on purpose."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None, env=env,
            timeout=timeout, capture_output=True,
        )
    except Exception as error:
        log.warning("archive git could not run: %s", describe(error))
        return False
    if done.returncode != 0:
        # git's own error text is the only place a rejected key says why, and
        # sanitise() strips anything shaped like a url or a token out of it.
        log.warning("archive git %s failed (%d): %s",
                    args[0], done.returncode,
                    sanitise(done.stderr.decode("utf-8", "replace"))[:400])
        return False
    return True


def checkout(log, root: Path, repo: str, key: str, days: list[str]) -> Path | None:
    """A working copy that has `days` materialised, or None if it could not be made.

    Returns the repository root. `days` are UTC date directories, which is what
    the writer stores under, not the local day being compiled.
    """
    env = git_env(root, key)
    work = root / "archive"
    if not git(log, env, "clone", "--quiet", "--depth", "1", "--filter=tree:0",
             "--sparse", f"git@{HOST}:{repo}.git", str(work)):
        return None
    paths = [f"{DIR}/{day}" for day in days]
    if paths and not git(log, env, "sparse-checkout", "set", *paths, cwd=work):
        return None
    return work


def commit(log, root: Path, repo: str, key: str, day: str, stem: str,
           text: str, metrics: dict, stamp: float) -> bool:
    """Write one transcript and its sidecar, commit, push. True only if it landed."""
    env = git_env(root, key)
    work = root / "archive"
    if not git(log, env, "clone", "--quiet", "--depth", "1", "--filter=tree:0",
               "--sparse", f"git@{HOST}:{repo}.git", str(work)):
        return False
    if not git(log, env, "sparse-checkout", "set", f"{DIR}/{day}", cwd=work):
        return False

    folder = work / DIR / day
    folder.mkdir(parents=True, exist_ok=True)
    # newline="\n" for the same reason as the key: the archived bytes must not
    # depend on the newline convention of the machine that happened to write
    # them, or the same transcript hashes differently on a laptop and a runner.
    (folder / f"{stem}.txt").write_text(text, encoding="utf-8", newline="\n")
    # The metrics ride alongside as a sidecar rather than in object metadata or a
    # commit message: a file is found by the same walk that finds the
    # transcripts, and a day is totalled without parsing a document header or a
    # git log. Every value is a string, because that is what the reader assumes.
    (folder / f"{stem}.json").write_text(
        json.dumps(
            {**{name: str(value) for name, value in metrics.items()},
             "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))},
            indent=1, sort_keys=True,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )

    if not git(log, env, "add", "--", f"{DIR}/{day}", cwd=work):
        return False
    # Identity per command rather than `git config`, so nothing is written into
    # a config file that outlives the temp directory.
    if not git(log, env, "-c", "user.name=media-utils-lab archive",
               "-c", "user.email=archive@users.noreply.github.com",
               "commit", "--quiet", "-m", f"transcript {day} {stem}", cwd=work):
        return False
    return git(log, env, "push", "--quiet", "origin", f"HEAD:{BRANCH}", cwd=work)
