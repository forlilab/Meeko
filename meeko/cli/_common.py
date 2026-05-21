"""Shared primitives used by both CLI entry points.

Originally these helpers lived inside ``mk_prepare_receptor.py``;
extracting them here lets the ligand CLI (and any future CLI) reuse
them without copy-paste.

The receptor module re-exports ``TalkativeParser``, ``check``, and
``required_length`` for backward compat so any third-party
``from meeko.cli.mk_prepare_receptor import TalkativeParser`` still
works.
"""

import argparse
import pathlib
import sys


def make_talkative_parser(script_path: pathlib.Path) -> type:
    """Build a ``TalkativeParser`` subclass that prints help + error
    line on parse failures. The script path is baked into the error
    message so the user sees something like ``mk_prepare_receptor.py:
    error: ...``.
    """

    class TalkativeParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            print(
                "\n%s: error: %s" % (script_path.name, message),
                file=sys.stderr,
            )
            sys.exit(2)

    return TalkativeParser


def check(success, error_msg):
    """Exit with code 2 if ``success`` is falsy."""
    if not success:
        print("Error: " + error_msg, file=sys.stderr)
        sys.exit(2)


def required_length(nmin: int, nmax: int):
    """Return an argparse Action that requires between ``nmin`` and
    ``nmax`` positional values."""

    class RequiredLength(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = "fargument {self.dest} requires between"
                msg += " {nmin} and {nmax} arguments"
                raise argparse.ArgumentTypeError(msg)
            setattr(namespace, self.dest, values)

    return RequiredLength
