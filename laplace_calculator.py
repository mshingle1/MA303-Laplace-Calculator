"""
╔══════════════════════════════════════════════════════════╗
║        Laplace Transform Calculator  (SymPy-powered)     ║
║  Supports: forward transform, inverse transform,         ║
║            symbolic input, step-by-step hints            ║
╚══════════════════════════════════════════════════════════╝

Run with:  python laplace_calculator.py
Requires:  pip install sympy
"""

import sys

# ── Dependency check ────────────────────────────────────────
try:
    import sympy as sp
    from sympy import (
        laplace_transform, inverse_laplace_transform,
        symbols, exp, sin, cos, sinh, cosh, Heaviside,
        DiracDelta, tan, atan, ln, sqrt, Abs, Piecewise,
        simplify, apart, pretty, latex, oo, pi, E, I,
        Function, Rational, factorial, gamma
    )
    from sympy.integrals.transforms import (
        LaplaceTransform, InverseLaplaceTransform
    )
except ImportError:
    print("\n  [!] SymPy not found. Install it with:\n\n      pip install sympy\n")
    sys.exit(1)

# ── Colour helpers (ANSI) ────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
DIM    = "\033[2m"

def c(colour, text):
    return f"{colour}{text}{RESET}"

# ── Symbols always available in user expressions ─────────────
t, s, a, b, n, k, w, omega = symbols("t s a b n k w omega", real=True, positive=True)
tau = symbols("tau", real=True, positive=True)

NAMESPACE = {
    # core SymPy
    "t": t, "s": s, "a": a, "b": b, "n": n, "k": k,
    "w": w, "omega": omega, "tau": tau,
    "pi": pi, "E": E, "I": I, "oo": oo,
    # functions
    "exp": exp, "sin": sin, "cos": cos,
    "sinh": sinh, "cosh": cosh,
    "tan": tan, "atan": atan,
    "ln": ln, "log": ln, "sqrt": sqrt,
    "Abs": Abs,
    "Heaviside": Heaviside,
    "DiracDelta": DiracDelta,
    "Piecewise": Piecewise,
    "factorial": factorial,
    "gamma": gamma,
    "Rational": Rational,
    "sqrt": sqrt,
    # allow t**n forms
    "t_pow": lambda p: t**p,
}

# ── Pretty header ─────────────────────────────────────────────
BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║         ℒ  Laplace Transform Calculator  ℒ⁻¹             ║
║                        MA303                             ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""

HELP_TEXT = f"""
{BOLD}Available symbols:{RESET}
  {CYAN}t, s{RESET}           — time / frequency variables
  {CYAN}a, b, k, n, w, omega, tau{RESET} — extra parameters
  {CYAN}pi, E, oo{RESET}      — π, Euler's number, ∞

{BOLD}Functions you can use:{RESET}
  {CYAN}exp(t){RESET}  {CYAN}sin(w*t){RESET}  {CYAN}cos(w*t){RESET}  {CYAN}sinh(a*t){RESET}  {CYAN}cosh(a*t){RESET}
  {CYAN}t**n{RESET}   (integer or rational n)
  {CYAN}Heaviside(t-a){RESET}   {CYAN}DiracDelta(t){RESET}
  {CYAN}ln(t){RESET}  {CYAN}sqrt(t){RESET}

{BOLD}Example expressions:{RESET}
  {YELLOW}t**2{RESET}                   → forward  → 2/s³
  {YELLOW}exp(-a*t)*sin(w*t){RESET}    → forward
  {YELLOW}1/(s*(s+1)){RESET}            → inverse  → 1 − e⁻ᵗ
  {YELLOW}s/(s**2 + w**2){RESET}        → inverse  → cos(w·t)
  {YELLOW}Heaviside(t-2)*exp(t){RESET} → forward

{BOLD}Commands:{RESET}
  {GREEN}L{RESET}   or {GREEN}forward{RESET}   — compute Laplace transform
  {GREEN}Li{RESET}  or {GREEN}inverse{RESET}   — compute inverse Laplace transform
  {GREEN}table{RESET}                 — show common transform pairs
  {GREEN}help{RESET}                  — show this message
  {GREEN}quit{RESET} / {GREEN}exit{RESET}        — exit
"""

COMMON_PAIRS = [
    ("1",                     "1/s",                     "s > 0"),
    ("t",                     "1/s²",                    "s > 0"),
    ("tⁿ  (n≥0 integer)",     "n! / s^(n+1)",            "s > 0"),
    ("exp(−a·t)",             "1/(s+a)",                  "s > −a"),
    ("t·exp(−a·t)",           "1/(s+a)²",                "s > −a"),
    ("sin(ω·t)",              "ω / (s²+ω²)",             "s > 0"),
    ("cos(ω·t)",              "s / (s²+ω²)",             "s > 0"),
    ("sinh(a·t)",             "a / (s²−a²)",             "s > |a|"),
    ("cosh(a·t)",             "s / (s²−a²)",             "s > |a|"),
    ("exp(−a·t)·sin(ω·t)",   "ω / ((s+a)²+ω²)",        "s > −a"),
    ("exp(−a·t)·cos(ω·t)",   "s+a / ((s+a)²+ω²)",      "s > −a"),
    ("δ(t)  [DiracDelta]",    "1",                        "all s"),
    ("u(t−a) [Heaviside]",   "e^(−as) / s",             "s > 0"),
    ("t^(1/2)",               "√π / (2·s^(3/2))",       "s > 0"),
    ("ln(t)",                 "−(γ + ln s)/s",           "s > 0"),
]


def print_table():
    col_w = [28, 26, 14]
    header = ["f(t)  (time domain)", "F(s)  (s domain)", "ROC"]
    sep = "─" * (sum(col_w) + 8)
    print(f"\n{CYAN}{BOLD}  Common Laplace Transform Pairs{RESET}")
    print(f"  {sep}")
    row = "  │ " + " │ ".join(h.ljust(col_w[i]) for i,h in enumerate(header)) + " │"
    print(c(BOLD, row))
    print(f"  {sep}")
    for ft, Fs, roc in COMMON_PAIRS:
        row = "  │ " + " │ ".join(
            [ft.ljust(col_w[0]), Fs.ljust(col_w[1]), roc.ljust(col_w[2])]
        ) + " │"
        print(row)
    print(f"  {sep}\n")


def parse_expr(raw: str):
    """Safely evaluate a SymPy expression string."""
    try:
        expr = sp.sympify(raw, locals=NAMESPACE)
        return expr, None
    except Exception as e:
        return None, str(e)


def do_laplace(expr):
    """Compute forward Laplace transform of expr(t)."""
    try:
        result = laplace_transform(expr, t, s, noconds=False)
        # result = (F(s), plane_of_convergence, conditions)
        if isinstance(result, tuple):
            F, plane, cond = result
        else:
            F, plane, cond = result, None, True
        F = simplify(F)
        return F, plane, cond
    except Exception as e:
        return None, None, str(e)


def do_inverse(expr):
    """Compute inverse Laplace transform of expr(s)."""
    try:
        # Try partial fractions first to help SymPy
        try:
            expr_pf = apart(expr, s)
        except Exception:
            expr_pf = expr
        result = inverse_laplace_transform(expr_pf, s, t, noconds=False)
        if isinstance(result, tuple):
            f, cond = result
        else:
            f, cond = result, True
        f = simplify(f)
        return f, cond
    except Exception as e:
        return None, str(e)


def display_result(label_in, expr_in, label_out, expr_out, extra=None):
    print()
    print(f"  {BOLD}Input  {label_in}:{RESET}")
    print(f"    {YELLOW}{pretty(expr_in)}{RESET}")
    print()
    print(f"  {BOLD}Result {label_out}:{RESET}")
    if expr_out is not None:
        print(f"    {GREEN}{pretty(expr_out)}{RESET}")
        print()
        print(f"  {DIM}LaTeX: {latex(expr_out)}{RESET}")
    if extra:
        print(f"  {MAGENTA}{extra}{RESET}")
    print()


def run():
    print(BANNER)
    print(f"  Type {c(GREEN,'help')} for usage, {c(GREEN,'table')} for common pairs, {c(GREEN,'quit')} to exit.\n")

    while True:
        try:
            cmd = input(c(CYAN, "  ℒ-calc » ") + RESET).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}  Goodbye!{RESET}\n")
            break

        if not cmd:
            continue

        low = cmd.lower()

        if low in ("quit", "exit", "q"):
            print(f"\n{DIM}  Goodbye!{RESET}\n")
            break

        if low in ("help", "h", "?"):
            print(HELP_TEXT)
            continue

        if low == "table":
            print_table()
            continue

        # ── Determine direction ──────────────────────────────
        if low.startswith("l ") or low.startswith("forward "):
            direction = "forward"
            raw = cmd.split(None, 1)[1].strip()
        elif low.startswith("li ") or low.startswith("inverse "):
            direction = "inverse"
            raw = cmd.split(None, 1)[1].strip()
        else:
            # Ask interactively
            print(f"\n  {BOLD}Direction?{RESET}  [{GREEN}L{RESET}] forward  [{GREEN}Li{RESET}] inverse  [{DIM}Enter to cancel{RESET}]")
            choice = input(f"  → ").strip().lower()
            if choice in ("l", "forward", "1"):
                direction = "forward"
            elif choice in ("li", "inverse", "2"):
                direction = "inverse"
            else:
                print(f"  {DIM}Cancelled.{RESET}\n")
                continue
            raw = cmd

        # ── Parse ────────────────────────────────────────────
        expr, err = parse_expr(raw)
        if err:
            print(f"\n  {RED}Parse error:{RESET} {err}")
            print(f"  {DIM}Check your expression and try again. Type 'help' for examples.{RESET}\n")
            continue

        # ── Compute ──────────────────────────────────────────
        if direction == "forward":
            F, plane, cond = do_laplace(expr)
            if F is None:
                print(f"\n  {RED}Could not compute Laplace transform.{RESET}")
                print(f"  {DIM}Reason: {cond}{RESET}")
                print(f"  {DIM}Try a simpler form or check the table.{RESET}\n")
            else:
                extra = None
                if plane is not None:
                    extra = f"Region of convergence: Re(s) > {plane}"
                if cond is not True and cond is not None and cond != True:
                    extra = (extra or "") + f"\n  Conditions: {cond}"
                display_result("f(t)", expr, "F(s)", F, extra)

        else:  # inverse
            f, cond = do_inverse(expr)
            if f is None:
                print(f"\n  {RED}Could not compute inverse Laplace transform.{RESET}")
                print(f"  {DIM}Reason: {cond}{RESET}")
                print(f"  {DIM}Try partial fraction form, e.g. 1/(s+1) + 2/(s+2){RESET}\n")
            else:
                extra = None
                if cond is not True and cond is not None and cond != True:
                    extra = f"Conditions: {cond}"
                display_result("F(s)", expr, "f(t)", f, extra)


if __name__ == "__main__":
    run()
