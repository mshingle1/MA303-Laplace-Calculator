# Graphing Tool for ℒaplace Transforms

A symbolic Laplace and inverse Laplace transform calculator for Python, powered by **SymPy** and **Matplotlib**. Computes exact symbolic results and automatically generates three-panel plots for every transform.

---

## Features

- **Forward transform** — `ℒ{ f(t) }` → `F(s)` with region of convergence
- **Inverse transform** — `ℒ⁻¹{ F(s) }` → `f(t)` with partial fraction decomposition
- **Exact symbolic output** — results are fully symbolic, never numerical approximations
- **LaTeX output** — every result is printed in LaTeX for easy copy-paste
- **3-panel plots** on every computation:
  - Forward: time-domain signal · frequency response (magnitude + phase) · 3D s-plane surface
  - Inverse: recovered signal · pole-zero map · partial fraction bar chart
- **Built-in reference table** of 14 common transform pairs
- **Togglable plotting** — run headless with the `noplot` command
- Coloured ANSI terminal UI with peak and steady-state annotations

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| sympy | any recent | Symbolic computation |
| numpy | any recent | Numerical evaluation |
| matplotlib | any recent | Plotting |

---

## Installation

```bash
pip install sympy numpy matplotlib
```

Then run:

```bash
python laplace_calculator.py
```

---

## Usage

Once running, type commands at the `ℒ-calc »` prompt.

### Forward Transform

```
L  <expression in t>
```

Computes `ℒ{ f(t) }` and opens a 3-panel plot.

```
L  t**2
L  exp(-a*t)*sin(w*t)
L  Heaviside(t-2)*cos(t)
```

### Inverse Transform

```
Li  <expression in s>
```

Computes `ℒ⁻¹{ F(s) }` and opens a 3-panel plot.

```
Li  1/(s*(s+1))
Li  s/(s**2 + 4)
Li  1/(s+1)**2
```

### Other Commands

| Command | Description |
|---------|-------------|
| `table` | Print common transform pairs |
| `noplot` | Toggle plotting on / off |
| `help` | Show usage and examples |
| `quit` | Exit |

---

## Supported Functions and Symbols

### Symbols

| Symbol | Meaning |
|--------|---------|
| `t` | Time variable |
| `s` | Complex frequency variable |
| `a`, `b`, `k`, `n` | Symbolic constants |
| `w`, `omega` | Angular frequency |
| `tau` | Time constant |
| `pi`, `E`, `oo` | π, Euler's number, ∞ |

### Functions

```
exp(x)    sin(x)    cos(x)    sinh(x)    cosh(x)
tan(x)    atan(x)   ln(x)     sqrt(x)    Abs(x)
Heaviside(t - a)    DiracDelta(t)
t**n  (any rational n)
```

---

## What the Plots Show

### Forward Transform — `L f(t)`

| Panel | Content |
|-------|---------|
| **f(t) — Time Domain** | Signal curve with filled area and peak annotation |
| **\|F(jω)\| — Frequency Response** | Magnitude (solid) and phase in degrees (dashed) on twin axes |
| **\|F(σ+jω)\| — s-Plane Surface** | 3D surface of \|F(s)\| over the complex plane |

The region of convergence is shown as a caption below the figure when available.

### Inverse Transform — `Li F(s)`

| Panel | Content |
|-------|---------|
| **f(t) — Recovered Signal** | Time-domain signal with steady-state estimate line |
| **Pole-Zero Map** | Poles marked `×`, zeros marked `○`, stable left-half-plane shaded |
| **Partial Fractions** | Colour-coded horizontal bar chart of each PFD term |

---

## Examples

```
ℒ-calc » L  t**3
  Result F(s):  6 / s⁴

ℒ-calc » L  exp(-t)*sin(2*t)
  Result F(s):  2 / ((s + 1)² + 4)

ℒ-calc » Li  1/(s*(s+1))
  Result f(t):  1 - e⁻ᵗ

ℒ-calc » Li  s/(s**2 + 9)
  Result f(t):  cos(3t)
```

Symbolic parameters like `a` and `w` are substituted with default numeric values for plotting only (`a=1`, `w=2`, etc.). The symbolic result displayed in the terminal is always fully exact.

---

## Limitations

- Only handles functions that SymPy can integrate analytically. Arbitrary piecewise or numerically-defined functions are not supported.
- The 3D s-plane surface plot is restricted to `Re(s) > 0` to avoid poles; functions with poles on the imaginary axis may show gaps.
- Very high-order rational functions may produce slow or incomplete partial fraction decompositions.

---

## File Structure

```
laplace_calculator.py   Main script — all logic in a single file
README.md               This file
```

---

## License

MIT — free to use, modify, and distribute.
