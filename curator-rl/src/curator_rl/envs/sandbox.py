"""Python code sandbox for the MBPP environment (Roadmap v3 Phase E, D.2; hardened by audit E-07).

Runs model-written code plus assertions in a separate interpreter process with a wall-clock
timeout, a memory and CPU limit (POSIX) and soft guards against network access and spawning
processes. Success is decided by a RANDOM per-run token printed only after every assertion has
executed, never by the exit code: code that calls `exit()` / `os._exit(0)` before the tests
produces no token and fails.

Reward-hack defences (audit E-07; an RL policy optimises exactly this reward):
1. The token never appears in the program file. The pre-audit version printed a literal
   sentinel at the end of the script, so `open(__file__)` + a regex + `os._exit(0)` scored. Now
   the token arrives on stdin, is read before any model code runs and lives only in a closure.
2. Static screening (`screen_solution`) rejects model code that reaches for the harness or the
   interpreter's internals: its own source (`__file__`, `open`), frames and globals
   (`globals`, `__globals__`, `f_globals`, `sys._getframe`, `gc`, `inspect`), the harness names
   (`_curator*`), dynamic `getattr` on non-literal names, and redefinitions of `__eq__` / `__ne__`
   / `__hash__`. A rejection is a failed verdict with its reason, so hacking attempts are logged.
3. Type-strict assertions: every `assert a == b` in the tests is rewritten to
   `assert _curator_eq(a, b)`, which refuses to let an object of a user-defined class equal a
   builtin value (the "always-equal object" hack passed every assertion before).
Gold MBPP solutions still pass 100% (contract test), so none of this rejects honest code.

THIS IS NOT A SECURITY BOUNDARY. In-process verification can never be made fully tamper-proof;
these layers stop the cheap, learnable hacks. Run it only on disposable machines (Kaggle
sessions, containers), never on a workstation with secrets. Cross-platform: the timeout and
token work on Windows; resource limits and process-group kill need POSIX (GPU runs are Linux).
"""

from __future__ import annotations

import ast
import os
import secrets
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass

_PREAMBLE = '''\
import socket as _socket, subprocess as _subprocess, os as _os
def _blocked(*_a, **_k):
    raise OSError("disabled in the sandbox")
for _n in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
    setattr(_socket, _n, _blocked)
for _n in ("Popen", "run", "call", "check_call", "check_output"):
    setattr(_subprocess, _n, _blocked)
for _n in ("system", "popen", "fork", "forkpty", "execv", "execve", "execl", "execvp", "spawnl", "spawnv",
           "posix_spawn", "posix_spawnp", "execle", "execlp", "execvpe", "spawnle", "spawnlp", "spawnve",
           "spawnvp", "spawnvpe"):
    if hasattr(_os, _n):
        setattr(_os, _n, _blocked)
del _socket, _subprocess, _os, _n
'''

# Read the token from stdin BEFORE model code runs; keep it (and every builtin the comparison
# needs) in a closure so later rebinding of globals such as `all` or `type` cannot touch them.
_HARNESS = '''\
def _curator_make():
    import sys as _sys
    _tok = _sys.stdin.readline().strip()
    _type, _len, _zip, _all, _print = type, len, zip, all, print
    _list, _tuple, _dict, _set, _fset = list, tuple, dict, set, frozenset
    _builtin = (int, float, bool, complex, str, bytes, list, tuple, dict, set, frozenset, type(None))
    _num = (int, float, bool, complex)

    def _eq(a, b):
        ta, tb = _type(a), _type(b)
        if ta not in _builtin or tb not in _builtin:
            return ta is tb and a == b          # a user object only equals one of its own class
        if ta in (_list, _tuple):
            return tb is ta and _len(a) == _len(b) and _all(_eq(x, y) for x, y in _zip(a, b))
        if ta is _dict:
            return tb is _dict and a.keys() == b.keys() and _all(_eq(a[k], b[k]) for k in a)
        if ta in (_set, _fset):
            return tb in (_set, _fset) and _all(_type(x) in _builtin for x in a) and a == b
        if ta in _num and tb in _num:
            return a == b
        return ta is tb and a == b

    def _done():
        _print(_tok, flush=True)

    return _eq, _done
_curator_eq, _curator_done = _curator_make()
del _curator_make
'''

# always refused: no honest MBPP solution needs them
_BANNED_ALWAYS = frozenset({"__file__", "__builtins__", "__loader__", "__spec__"})
# refused only when they mean the builtin: a solution may well call a parameter `input` (mbpp-644)
_BANNED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
})
_BANNED_ATTRS = frozenset({
    "__globals__", "__closure__", "__code__", "__dict__", "__subclasses__", "__bases__", "__mro__",
    "f_globals", "f_locals", "f_back", "f_builtins", "tb_frame", "gi_frame", "cr_frame",
    "ag_frame", "cell_contents", "_getframe", "settrace", "setprofile", "modules", "environ",
    "getenv", "__builtins__",
})
_BANNED_MODULES = frozenset({"gc", "inspect", "ctypes", "importlib", "builtins", "__main__",
                             "signal", "atexit", "pty", "multiprocessing"})
_BANNED_DEFS = frozenset({"__eq__", "__ne__", "__hash__", "__instancecheck__", "__subclasscheck__"})


def screen_solution(code: str) -> str | None:
    """Reason the code is refused, or None. Rejects reaches for the harness or interpreter internals."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return None                                   # let it run and fail as a syntax error
    bound = _bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (
                node.id in _BANNED_ALWAYS or node.id.startswith("_curator")
                or (node.id in _BANNED_BUILTINS and node.id not in bound)):
            return f"banned name '{node.id}'"
        if isinstance(node, ast.Attribute) and (node.attr in _BANNED_ATTRS or node.attr.startswith("_curator")):
            return f"banned attribute '.{node.attr}'"
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for mod in mods:
                if mod.split(".")[0] in _BANNED_MODULES:
                    return f"banned module '{mod}'"
            if isinstance(node, ast.ImportFrom) and any(a.name in _BANNED_ATTRS for a in node.names):
                return "banned import of an interpreter internal"
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _BANNED_DEFS:
            return f"redefines '{node.name}'"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "getattr", "setattr", "delattr", "hasattr"):
            if len(node.args) < 2 or not (isinstance(node.args[1], ast.Constant)
                                          and isinstance(node.args[1].value, str)):
                return f"dynamic {node.func.id}"
            name = node.args[1].value
            if name in _BANNED_ATTRS or name in _BANNED_DEFS or name.startswith("_curator"):
                return f"{node.func.id} of '{name}'"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                "/proc" in node.value or "_curator" in node.value):
            return "banned string literal"
    return None


def _bound_names(tree: ast.AST) -> set[str]:
    """Names the solution binds itself (parameters, assignments, defs, imports, loop targets)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            out.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[0])
    return out


def strict_assertions(test: str) -> str:
    """Rewrite top-level `assert a == b` into `assert _curator_eq(a, b)` (keeps the message)."""
    try:
        with warnings.catch_warnings():          # MBPP tests contain e.g. '\.' in strings
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(test)
    except SyntaxError:
        return test
    changed = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)):
            node.test = ast.Call(func=ast.Name(id="_curator_eq", ctx=ast.Load()),
                                 args=[node.test.left, node.test.comparators[0]], keywords=[])
            changed = True
    return ast.unparse(ast.fix_missing_locations(tree)) if changed else test


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    timed_out: bool
    seconds: float
    returncode: int | None
    stderr_tail: str
    rejected: str | None = None      # screening reason when the code was refused without running


def _limits_preamble(timeout_s: float, memory_mb: int | None) -> str:
    """Resource limits applied by the child interpreter itself, before any model code runs.

    Set inside the child (not via `preexec_fn`) so `run_python_tests` is safe to call from a
    thread pool: `preexec_fn` is unsafe in a multi-threaded parent (it can deadlock after fork).
    """
    if os.name != "posix":
        return ""
    cpu = (int(timeout_s) + 1, int(timeout_s) + 2)
    fsize = 64 * 1024 * 1024
    lines = [
        "import resource as _resource",
        f"_resource.setrlimit(_resource.RLIMIT_CPU, {cpu!r})",
        f"_resource.setrlimit(_resource.RLIMIT_FSIZE, ({fsize}, {fsize}))",
    ]
    if memory_mb:
        cap = memory_mb * 1024 * 1024
        lines.append(f"_resource.setrlimit(_resource.RLIMIT_AS, ({cap}, {cap}))")
    lines.append("del _resource")
    return "\n".join(lines) + "\n"


def run_python_tests(
    code: str,
    tests: list[str],
    setup: str = "",
    *,
    timeout_s: float = 5.0,
    memory_mb: int | None = 512,
) -> SandboxResult:
    """Execute `code`, then `setup`, then each assertion in `tests`; pass iff the token appears.

    Setup runs AFTER the solution (standard MBPP harness order): some tasks build test objects from a
    class the solution defines (e.g. mbpp-927). Thread-safe: many calls may run concurrently.
    """
    start = time.perf_counter()
    reason = screen_solution(code)
    if reason is not None:
        return SandboxResult(passed=False, timed_out=False, seconds=time.perf_counter() - start,
                             returncode=None, stderr_tail="", rejected=reason)
    token = "SBX" + secrets.token_hex(16)
    preamble = _limits_preamble(timeout_s, memory_mb) + _PREAMBLE + _HARNESS
    checks = [strict_assertions(t) for t in tests]
    program = "\n".join([preamble, code, "", setup or "", "", *checks, "_curator_done()", ""])
    with tempfile.TemporaryDirectory(prefix="curator_sbx_") as tmp:
        script = os.path.join(tmp, "prog.py")
        out_path = os.path.join(tmp, "out.txt")
        err_path = os.path.join(tmp, "err.txt")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                proc = subprocess.run(
                    [sys.executable, "-I", script], stdout=out, stderr=err, input=(token + "\n").encode(),
                    cwd=tmp, env=env, timeout=timeout_s, start_new_session=(os.name == "posix"),
                    check=False,
                )
            timed_out, rc = False, proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, None
        stdout = _tail(out_path)
        stderr = _tail(err_path)
    passed = (not timed_out) and rc == 0 and token in stdout
    return SandboxResult(passed=passed, timed_out=timed_out, seconds=time.perf_counter() - start,
                         returncode=rc, stderr_tail=stderr[-400:])


def _tail(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(size - limit, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
