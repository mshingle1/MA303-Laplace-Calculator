"""
Laplace Transform Graphing Calculator          

Run:      python laplace_calculator.py
Requires: pip install sympy matplotlib numpy
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def _require(pkg, install):

    try:
        return __import__(pkg)
    except ImportError:
        print(f"\n  [!] '{pkg}' not found.  pip install {install}\n")
        sys.exit(1)
_require("sympy",      "sympy")
_require("numpy",      "numpy")
_require("matplotlib", "matplotlib")
#----------------------------------------------------------------FORMATTING--------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sympy as sp
from sympy import (
    laplace_transform, inverse_laplace_transform,
    symbols, exp, sin, cos, sinh, cosh, Heaviside,
    DiracDelta, tan, atan, ln, sqrt, Abs, Piecewise,
    simplify, apart, pretty, latex, oo, pi, E, I,
    Function, Rational, factorial, gamma,
)

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
DIM     = "\033[2m"

def c(colour, text):
    return f"{colour}{text}{RESET}"

t, s, a, b, n, k, w, omega = symbols("t s a b n k w omega",
                                      real=True, positive=True)
tau = symbols("tau", real=True, positive=True)

NAMESPACE = {
    "t": t, "s": s, "a": a, "b": b, "n": n, "k": k,
    "w": w, "omega": omega, "tau": tau,
    "pi": pi, "E": E, "I": I, "oo": oo,
    "exp": exp, "sin": sin, "cos": cos,
    "sinh": sinh, "cosh": cosh,
    "tan": tan, "atan": atan,
    "ln": ln, "log": ln, "sqrt": sqrt,
    "Abs": Abs, "Heaviside": Heaviside,
    "DiracDelta": DiracDelta, "Piecewise": Piecewise,
    "factorial": factorial, "gamma": gamma,
    "Rational": Rational,
}

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#58a6ff",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "font.family":       "monospace",
    "figure.titlesize":  13,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.framealpha": 0.2,
    "legend.fontsize":   8,
})

C_TIME  = "#58a6ff"   # blue   — time domain
C_FREQ  = "#3fb950"   # green  — frequency magnitude
C_PHASE = "#d2a8ff"   # purple — phase
C_ANNOT = "#f0883e"   # orange — annotations
C_POLE  = "#ff7b72"   # red    — poles
C_ZERO  = "#58a6ff"   # blue   — zeros
C_CMAP  = "plasma"    # colormap for 3-D surface
#----------------------------------------------------------------CALCULATIONS-----------------------------------------------------------------------------------------


def parse_expr(raw: str):
    try:
        expr = sp.sympify(raw, locals=NAMESPACE)
        return expr, None
    except Exception as e:
        return None, str(e)


def do_laplace(expr):
    try:
        result = laplace_transform(expr, t, s, noconds=False)
        F, plane, cond = (result if isinstance(result, tuple)
                          else (result, None, True))
        return simplify(F), plane, cond
    except Exception as e:
        return None, None, str(e)


def do_inverse(expr):
    try:
        try:
            expr_pf = apart(expr, s)
        except Exception:
            expr_pf = expr
        result = inverse_laplace_transform(expr_pf, s, t, noconds=False)
        f, cond = (result if isinstance(result, tuple) else (result, True))
        return simplify(f), cond
    except Exception as e:
        return None, str(e)


def _default_subs(expr):
    """Return nice numeric substitutions for free symbolic parameters."""
    free = {str(v) for v in expr.free_symbols} - {"t", "s"}
    nice = {"a": 1.0, "b": 2.0, "k": 1.0, "n": 2.0,
            "w": 2.0, "omega": 2.0, "tau": 1.0}
    return {v: nice.get(v, 1.0) for v in free}


def lambdify_safe(sym_var, expr, subs=None):
    local = expr
    if subs:
        local = expr.subs({sp.Symbol(k): v for k, v in subs.items()})
    try:
        return sp.lambdify(sym_var, local, modules=["numpy"])
    except Exception:
        return None


def safe_eval(fn, xs):
    try:
        ys = fn(xs)
        if np.isscalar(ys):
            ys = np.full_like(xs, float(ys), dtype=float)
        ys = np.array(ys, dtype=complex)
        ys = np.where(np.isfinite(ys), ys, np.nan + 0j)
        return ys.real
    except Exception:
        return np.full_like(xs, np.nan, dtype=float)


#-----------------------------------------------------------------Plotting------------------------------------------------------------------------------------
#  FORWARD TRANSFORM — 3-panel plot
#
#  Panel 1  f(t)          — time-domain signal with fill + peak annotation
#  Panel 2  |F(jω)|       — frequency magnitude & phase (twin axes)
#  Panel 3  |F(σ+jω)|     — 3-D surface over the s-plane

def plot_forward(ft_expr, Fs_expr, ft_str, Fs_str, roc=None):
    subs = _default_subs(ft_expr)

    # ── numerical lambdas ─────────────────────────────────────
    ft_fn = lambdify_safe(t, ft_expr, subs)

    s_sym = sp.Symbol("s_num")
    Fs_num = Fs_expr.subs({sp.Symbol(k): v for k, v in subs.items()})
    try:
        Fs_fn = sp.lambdify(s_sym, Fs_num.subs(s, s_sym),
                            modules=["numpy"])
    except Exception:
        Fs_fn = None

    t_vals  = np.linspace(0, 8, 1000)
    om_vals = np.linspace(0, 20, 800)
    ft_vals = safe_eval(ft_fn, t_vals) if ft_fn else None

    mag = phase = None
    if Fs_fn is not None:
        try:
            jom  = 1j * om_vals
            Fjom = Fs_fn(jom)
            Fjom = np.where(np.isfinite(Fjom), Fjom, np.nan + 0j)
            mag   = np.abs(Fjom)
            phase = np.angle(Fjom, deg=True)
        except Exception:
            pass

    # 3-D grid
    sig_vals = np.linspace(0.05, 1.5, 55)
    om3_vals = np.linspace(-10, 10, 55)
    SIG, OM  = np.meshgrid(sig_vals, om3_vals)
    Z = None
    if Fs_fn is not None:
        try:
            Zraw = Fs_fn(SIG + 1j * OM)
            Zraw = np.abs(Zraw)
            Z    = np.where(np.isfinite(Zraw) & (Zraw < 1e4), Zraw, np.nan)
        except Exception:
            pass

    # ── figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5.6), facecolor="#0d1117")
    fig.suptitle(
        f"ℒ{{  {ft_str}  }}  =  {Fs_str}",
        fontsize=12, color=C_TIME, fontweight="bold", y=1.02,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

    # ── Panel 1 : f(t) ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("f(t)  —  Time Domain")
    ax1.set_xlabel("t")
    ax1.set_ylabel("f(t)")
    if ft_vals is not None:
        ax1.fill_between(t_vals, ft_vals, alpha=0.14, color=C_TIME)
        ax1.plot(t_vals, ft_vals, color=C_TIME, lw=2.0, label="f(t)")
        ax1.axhline(0, color="#30363d", lw=0.8)
        ax1.axvline(0, color="#30363d", lw=0.8)
        finite = ft_vals[np.isfinite(ft_vals)]
        if len(finite):
            pk_i = int(np.argmax(np.abs(finite)))
            ax1.annotate(
                f"peak ≈ {finite[pk_i]:.2f}",
                xy=(t_vals[pk_i], finite[pk_i]),
                xytext=(t_vals[pk_i] + 0.6, finite[pk_i] * 0.80),
                arrowprops=dict(arrowstyle="->", color=C_ANNOT),
                color=C_ANNOT, fontsize=7,
            )
    else:
        ax1.text(0.5, 0.5, "Cannot plot\n(symbolic/complex)",
                 ha="center", va="center", transform=ax1.transAxes,
                 color="#8b949e")
    ax1.legend()

    # ── Panel 2 : |F(jω)| magnitude + phase ──────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("|F(jω)|  —  Frequency Response")
    ax2.set_xlabel("ω  (rad/s)")
    ax2.set_ylabel("|F(jω)|", color=C_FREQ)
    ax2.tick_params(axis="y", labelcolor=C_FREQ)
    if mag is not None:
        ax2.fill_between(om_vals, mag, alpha=0.14, color=C_FREQ)
        ax2.plot(om_vals, mag, color=C_FREQ, lw=2.0, label="|F(jω)|")
        ax2b = ax2.twinx()
        ax2b.set_facecolor("#161b22")
        ax2b.plot(om_vals, phase, color=C_PHASE, lw=1.3,
                  ls="--", alpha=0.85, label="∠F(jω)")
        ax2b.set_ylabel("Phase  (°)", color=C_PHASE)
        ax2b.tick_params(axis="y", labelcolor=C_PHASE)
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "Cannot evaluate\nF(jω) numerically",
                 ha="center", va="center", transform=ax2.transAxes,
                 color="#8b949e")

    # ── Panel 3 : 3-D s-plane surface ────────────────────────
    ax3 = fig.add_subplot(gs[2], projection="3d")
    ax3.set_facecolor("#0d1117")
    ax3.set_title("|F(σ+jω)|  —  s-Plane Surface")
    ax3.set_xlabel("σ = Re(s)", labelpad=5)
    ax3.set_ylabel("ω = Im(s)", labelpad=5)
    ax3.set_zlabel("|F(s)|",    labelpad=5)
    if Z is not None:
        surf = ax3.plot_surface(SIG, OM, Z, cmap=C_CMAP,
                                alpha=0.88, linewidth=0,
                                antialiased=True)
        fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=12,
                     pad=0.1, label="|F(s)|")
        # jω axis marker
        ax3.plot([0]*40, np.linspace(-10, 10, 40), np.zeros(40),
                 color="white", lw=0.8, ls=":", alpha=0.5)
    else:
        ax3.text2D(0.5, 0.5, "s-plane surface\nnot available",
                   ha="center", va="center", transform=ax3.transAxes,
                   color="#8b949e")

    if roc is not None:
        fig.text(0.5, -0.03,
                 f"Region of Convergence :  Re(s) > {roc}",
                 ha="center", fontsize=9, color=C_PHASE)

    plt.tight_layout()
    plt.show()


#  INVERSE TRANSFORM — 3-panel plot
#
#  Panel 1  f(t)          — recovered time signal with steady-state line
#  Panel 2  Pole-zero map — poles (×) and zeros (○) in s-plane
#  Panel 3  Partial fractions — colour-coded term breakdown

def plot_inverse(Fs_expr, ft_expr, Fs_str, ft_str):
    subs   = _default_subs(Fs_expr)
    sym_sub = {sp.Symbol(k): v for k, v in subs.items()}

    Fs_num = Fs_expr.subs(sym_sub)
    ft_num = ft_expr.subs(sym_sub)

    t_vals  = np.linspace(0, 12, 1400)
    ft_fn   = lambdify_safe(t, ft_num, {})
    ft_vals = safe_eval(ft_fn, t_vals) if ft_fn else None

    # ── poles & zeros ─────────────────────────────────────────
    poles, zeros = [], []
    pf_terms = []
    try:
        numer, denom = sp.fraction(sp.together(Fs_num))
        for p in sp.solve(denom, s):
            try:
                poles.append(complex(p.evalf()))
            except Exception:
                pass
        for z in sp.solve(numer, s):
            try:
                zeros.append(complex(z.evalf()))
            except Exception:
                pass
    except Exception:
        pass

    try:
        pf_expr = apart(Fs_num, s)
        pf_terms = [str(term) for term in sp.Add.make_args(pf_expr)]
    except Exception:
        pass

    # ── figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5.6), facecolor="#0d1117")
    fig.suptitle(
        f"ℒ⁻¹{{  {Fs_str}  }}  =  {ft_str}",
        fontsize=12, color=C_FREQ, fontweight="bold", y=1.02,
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # ── Panel 1 : f(t) ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("f(t)  —  Recovered Time Signal")
    ax1.set_xlabel("t")
    ax1.set_ylabel("f(t)")
    if ft_vals is not None:
        ax1.fill_between(t_vals, ft_vals, alpha=0.14, color=C_TIME)
        ax1.plot(t_vals, ft_vals, color=C_TIME, lw=2.0, label="f(t)")
        ax1.axhline(0, color="#30363d", lw=0.8)
        ax1.axvline(0, color="#30363d", lw=0.8)
        # steady-state estimate
        tail = ft_vals[np.isfinite(ft_vals)]
        if len(tail) > 200:
            ss = float(np.mean(tail[-200:]))
            if abs(ss) < 1e4:
                ax1.axhline(ss, color=C_ANNOT, lw=1.1,
                            ls=":", alpha=0.85)
                ax1.text(t_vals[-1] * 0.55, ss * 1.06,
                         f"ss ≈ {ss:.3f}",
                         color=C_ANNOT, fontsize=7)
    else:
        ax1.text(0.5, 0.5, "Cannot plot\n(symbolic/complex)",
                 ha="center", va="center", transform=ax1.transAxes,
                 color="#8b949e")
    ax1.legend()

    # ── Panel 2 : Pole-zero map ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Pole-Zero Map  of  F(s)")
    ax2.set_xlabel("Re(s)  →  σ")
    ax2.set_ylabel("Im(s)  →  jω")
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.axvline(0, color="#30363d", lw=1.2, ls="--", alpha=0.5)

    if poles:
        ax2.scatter([p.real for p in poles], [p.imag for p in poles],
                    marker="x", s=130, color=C_POLE, lw=2.5,
                    zorder=5, label="Poles  ×")
        for p in poles:
            ax2.annotate(
                f"  {p.real:.2f}{'+' if p.imag >= 0 else ''}{p.imag:.2f}j",
                xy=(p.real, p.imag), fontsize=7, color=C_POLE,
            )

    if zeros:
        ax2.scatter([z.real for z in zeros], [z.imag for z in zeros],
                    marker="o", s=110, facecolors="none",
                    edgecolors=C_ZERO, lw=2.0, zorder=5, label="Zeros  ○")
        for z in zeros:
            ax2.annotate(
                f"  {z.real:.2f}{'+' if z.imag >= 0 else ''}{z.imag:.2f}j",
                xy=(z.real, z.imag), fontsize=7, color=C_ZERO,
            )

    if not poles and not zeros:
        ax2.text(0.5, 0.5, "No finite poles/zeros\nfound",
                 ha="center", va="center", transform=ax2.transAxes,
                 color="#8b949e")

    # shade the left-half plane (stable region)
    xl = ax2.get_xlim()
    yl = ax2.get_ylim()
    ax2.axvspan(min(xl[0], -2), 0, alpha=0.07,
                color=C_FREQ, label="LHP (stable)")
    ax2.set_xlim(xl)
    ax2.set_ylim(yl)
    ax2.legend(loc="upper right")

    # ── Panel 3 : Partial fraction decomposition ──────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("Partial Fraction Decomposition")
    BAR_COLS = [C_TIME, C_FREQ, C_PHASE, C_ANNOT, "#ffa657", C_POLE]
    if pf_terms:
        y_pos = list(range(len(pf_terms)))
        ax3.barh(
            y_pos,
            [1] * len(pf_terms),
            color=[BAR_COLS[i % len(BAR_COLS)] for i in y_pos],
            alpha=0.72, height=0.55,
        )
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"  {tr}" for tr in pf_terms], fontsize=8)
        ax3.set_xticks([])
        ax3.invert_yaxis()
        ax3.set_xlabel("F(s)  =  Σ  (terms listed)", fontsize=8)
        # total as footer
        ax3.text(0.98, -0.14,
                 "F(s) = " + " + ".join(pf_terms),
                 transform=ax3.transAxes, fontsize=6.5,
                 ha="right", color="#8b949e")
    else:
        ax3.text(0.5, 0.5,
                 "Partial fraction\ndecomposition\nnot available",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="#8b949e")
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.tight_layout()
    plt.show()

#------------------------------------------------terminal formatting-------------------------------------------------------------------------------------

BANNER = f"""
{YELLOW}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                             ℒaplace Transform Graphing Calculator                                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝{RESET}
"""

HELP_TEXT = f"""
{BOLD}Symbols:{RESET}
  {CYAN}t, s{RESET}  — time / frequency        {CYAN}a, b, k, n, w, omega, tau{RESET}
  {CYAN}pi, E, oo{RESET} — π, e, ∞

{BOLD}Functions:{RESET}
  {CYAN}exp  sin  cos  sinh  cosh  tan  atan  ln  sqrt
  Heaviside(t-a)   DiracDelta(t){RESET}

{BOLD}Examples:{RESET}
  {YELLOW}L  t**2{RESET}                  → forward → 2/s³
  {YELLOW}L  exp(-a*t)*sin(w*t){RESET}   → forward
  {YELLOW}Li 1/(s*(s+1)){RESET}           → inverse → 1 − e⁻ᵗ
  {YELLOW}Li s/(s**2 + 4){RESET}          → inverse → cos(2t)

{BOLD}Commands:{RESET}
  {GREEN}L  <expr>{RESET}   forward Laplace transform + 3-panel plot
  {GREEN}Li <expr>{RESET}   inverse Laplace transform + 3-panel plot
  {GREEN}table{RESET}       common transform pairs
  {GREEN}noplot{RESET}      toggle plotting on/off
  {MAGENTA}help{RESET}        this message
  {MAGENTA}quit{RESET}        exit

{DIM}Symbolic parameters (a, w, …) are substituted with nice defaults
for plotting only — the symbolic result is always exact.{RESET}
"""

COMMON_PAIRS = [
    ("1",                  "1/s",               "s > 0"),
    ("t",                  "1/s²",              "s > 0"),
    ("tⁿ",                "n!/s^(n+1)",        "s > 0"),
    ("exp(−at)",           "1/(s+a)",           "s > −a"),
    ("t·exp(−at)",         "1/(s+a)²",         "s > −a"),
    ("sin(ωt)",            "ω/(s²+ω²)",        "s > 0"),
    ("cos(ωt)",            "s/(s²+ω²)",        "s > 0"),
    ("sinh(at)",           "a/(s²−a²)",        "s > |a|"),
    ("cosh(at)",           "s/(s²−a²)",        "s > |a|"),
    ("exp(−at)sin(ωt)",    "ω/((s+a)²+ω²)",   "s > −a"),
    ("exp(−at)cos(ωt)",    "(s+a)/((s+a)²+ω²)","s > −a"),
    ("δ(t)  DiracDelta",   "1",                 "all s"),
    ("u(t−a) Heaviside",   "e^(−as)/s",        "s > 0"),
    ("√t",                 "√π/(2s^(3/2))",    "s > 0"),
]


def print_table():
    col_w = [26, 24, 12]
    sep = "─" * (sum(col_w) + 8)
    print(f"\n{CYAN}{BOLD}  Common Laplace Transform Pairs{RESET}")
    print(f"  {sep}")
    hdr = ["f(t)", "F(s)", "ROC"]
    print(c(BOLD, "  │ " +
            " │ ".join(h.ljust(col_w[i]) for i, h in enumerate(hdr)) +
            " │"))
    print(f"  {sep}")
    for ft_, Fs_, roc_ in COMMON_PAIRS:
        print("  │ " +
              " │ ".join([ft_.ljust(col_w[0]),
                          Fs_.ljust(col_w[1]),
                          roc_.ljust(col_w[2])]) +
              " │")
    print(f"  {sep}\n")


def display_result(label_in, expr_in, label_out, expr_out, extra=None):
    print()
    print(f"  {BOLD}Input  {label_in}:{RESET}")
    print(f"    {YELLOW}{sp.pretty(expr_in)}{RESET}")
    print()
    print(f"  {BOLD}Result {label_out}:{RESET}")
    if expr_out is not None:
        print(f"    {GREEN}{sp.pretty(expr_out)}{RESET}")
        print()
        print(f"  {DIM}LaTeX: {sp.latex(expr_out)}{RESET}")
    if extra:
        print(f"  {MAGENTA}{extra}{RESET}")
    print()


def run():
    plotting = True
    print(BANNER)
    print(f"  Type {c(GREEN,'help')} · {c(GREEN,'table')} · {c(GREEN,'quit')}"
          f"      Plotting: {c(GREEN,'ON')}\n")

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
        if low == "noplot":
            plotting = not plotting
            state = c(GREEN, "ON") if plotting else c(RED, "OFF")
            print(f"  Plotting is now {state}\n")
            continue

        # ── Direction ──────────────────────────────────────────
        if low.startswith("l ") or low.startswith("forward "):
            direction = "forward"
            raw = cmd.split(None, 1)[1].strip()
        elif low.startswith("li ") or low.startswith("inverse "):
            direction = "inverse"
            raw = cmd.split(None, 1)[1].strip()
        else:
            print(f"\n  {BOLD}Direction?{RESET}  "
                  f"[{GREEN}L{RESET}] forward  "
                  f"[{GREEN}Li{RESET}] inverse  "
                  f"[{DIM}Enter = cancel{RESET}]")
            choice = input("  → ").strip().lower()
            if choice in ("l", "forward", "1"):
                direction = "forward"
            elif choice in ("li", "inverse", "2"):
                direction = "inverse"
            else:
                print(f"  {DIM}Cancelled.{RESET}\n")
                continue
            raw = cmd

        # ── Parse ──────────────────────────────────────────────
        expr, err = parse_expr(raw)
        if err:
            print(f"\n  {RED}Parse error:{RESET} {err}")
            print(f"  {DIM}Type 'help' for examples.{RESET}\n")
            continue

        # ── Compute & display ──────────────────────────────────
        if direction == "forward":
            F, plane, cond = do_laplace(expr)
            if F is None:
                print(f"\n  {RED}Could not compute Laplace transform.{RESET}")
                print(f"  {DIM}Reason: {cond}{RESET}\n")
            else:
                extra = (f"ROC: Re(s) > {plane}" if plane is not None
                         else None)
                display_result("f(t)", expr, "F(s)", F, extra)
                if plotting:
                    print(f"  {DIM}Opening plot…{RESET}\n")
                    try:
                        plot_forward(expr, F,
                                     ft_str=str(expr),
                                     Fs_str=str(F),
                                     roc=plane)
                    except Exception as e:
                        print(f"  {RED}Plot error:{RESET} {e}\n")

        else:
            f, cond = do_inverse(expr)
            if f is None:
                print(f"\n  {RED}Could not compute inverse Laplace "
                      f"transform.{RESET}")
                print(f"  {DIM}Reason: {cond}{RESET}\n")
            else:
                extra = (None if cond is True else f"Conditions: {cond}")
                display_result("F(s)", expr, "f(t)", f, extra)
                if plotting:
                    print(f"  {DIM}Opening plot…{RESET}\n")
                    try:
                        plot_inverse(expr, f,
                                     Fs_str=str(expr),
                                     ft_str=str(f))
                    except Exception as e:
                        print(f"  {RED}Plot error:{RESET} {e}\n")


if __name__ == "__main__":
    run()
