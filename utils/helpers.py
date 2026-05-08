"""tqdm-aware printing, progress bars, and CLI/config-file overrides."""

import os
import sys
from ast import literal_eval

from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# Console / progress helpers
# ----------------------------------------------------------------------

def console_quiet() -> bool:
    mode = os.environ.get("TENSORCACHE_QUIET", "0").strip().lower()
    return mode in {"1", "on", "true", "yes", "quiet"}


def progress_enabled() -> bool:
    mode = os.environ.get("TENSORCACHE_TQDM", "auto").strip().lower()
    if mode in {"0", "off", "false", "disable", "disabled"}:
        return False
    if mode in {"1", "on", "true", "force"}:
        return True
    return sys.stderr.isatty()


def _base_position() -> int:
    raw = os.environ.get("TENSORCACHE_TQDM_POSITION", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _prefixed_desc(desc: str) -> str:
    prefix = os.environ.get("TENSORCACHE_TQDM_DESC_PREFIX", "").strip()
    if prefix and desc:
        return f"{prefix} {desc}"
    if prefix:
        return prefix
    return desc


def _ascii_mode() -> bool:
    mode = os.environ.get("TENSORCACHE_TQDM_ASCII", "").strip().lower()
    return mode in {"1", "on", "true", "force", "yes"}


def make_progress(iterable=None, *, total=None, desc="", position_offset=0,
                  leave=True, disable=None, **kwargs):
    if disable is None:
        disable = not progress_enabled()
    mininterval = float(os.environ.get("TENSORCACHE_TQDM_MININTERVAL", "0.5"))
    return tqdm(
        iterable,
        total=total,
        desc=_prefixed_desc(desc),
        position=_base_position() + int(position_offset),
        leave=leave,
        disable=disable,
        dynamic_ncols=True,
        ascii=_ascii_mode(),
        mininterval=mininterval,
        smoothing=0.05,
        **kwargs,
    )


def tprint(msg="", end="\n"):
    """Print a message that coexists cleanly with any active tqdm bar."""
    tqdm.write(str(msg), end=end)


def cprint(*args, force=False, **kwargs):
    """Print only when console quiet mode is disabled unless force=True."""
    if force or not console_quiet():
        print(*args, **kwargs)


def ctprint(msg="", end="\n", force=False):
    """tqdm-aware print honoring console quiet mode unless force=True."""
    if force or not console_quiet():
        tprint(msg, end=end)


# ----------------------------------------------------------------------
# Config-file / CLI override system
#
# Karpathy-style: scripts declare module-level variables, then call
# apply_overrides(globals()) to absorb positional config files and --key=value
# CLI flags.
# ----------------------------------------------------------------------

def _config_quiet() -> bool:
    mode = os.environ.get("TENSORCACHE_CONFIG_QUIET", "").strip().lower()
    if mode:
        return mode in {"1", "on", "true", "yes", "quiet"}
    return console_quiet()


def apply_overrides(g):
    """Apply positional config files and --key=value CLI overrides to globals dict `g`."""
    quiet = _config_quiet()
    for arg in sys.argv[1:]:
        if "=" not in arg:
            assert not arg.startswith("--"), f"Unexpected flag without value: {arg}"
            config_file = arg
            if not quiet:
                print(f"Overriding config with {config_file}:")
                with open(config_file) as f:
                    print(f.read())
            with open(config_file) as f:
                exec(f.read(), g)
        else:
            assert arg.startswith("--"), f"Expected --key=value, got: {arg}"
            key, val = arg.split("=", 1)
            key = key[2:]
            if key not in g:
                raise ValueError(f"Unknown config key: {key}")
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            current = g[key]
            if type(attempt) is not type(current):
                if isinstance(current, str):
                    attempt = val
                elif isinstance(current, float) and isinstance(attempt, int):
                    attempt = float(attempt)
                else:
                    raise TypeError(
                        f"Config key '{key}': type mismatch, expected "
                        f"{type(current).__name__} but got "
                        f"{type(attempt).__name__} from value '{val}'"
                    )
            if not quiet:
                print(f"Overriding: {key} = {attempt}")
            g[key] = attempt
