# Graphing Tool for ℒaplace Transforms

A symbolic Laplace and inverse Laplace transform calculator for Python, powered by **SymPy** and **Matplotlib**. Computes exact symbolic results and automatically generates three-panel plots for every transform.


## Features

- **Forward transform** — with region of convergence graph
- **Inverse transform** — partial fraction decomposition graph
- Reference table for common transforms
- Plotting
  - Forward: time-domain signal · frequency response (magnitude + phase) · 3D s-plane surface
  - Inverse: recovered signal · pole-zero map · partial fraction bar chart

| Command | Description |
|---------|-------------|
| `table` | Print common transform pairs |
| `noplot` | Toggle plotting on / off |
| `help` | Show usage and examples |
| `quit` | Exit |

## Requirements
- Be sure to install sympy, numpy, and matplotlib in your terminal


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

## Plotting

### Forward Transform

| Panel | Content |
|-------|---------|
| **f(t) — Time Domain** | Signal curve with filled area and peak annotation |
| **\|F(jω)\| — Frequency Response** | Magnitude (solid) and phase in degrees (dashed) on twin axes |
| **\|F(σ+jω)\| — s-Plane Surface** | 3D surface of \|F(s)\| over the complex plane |


### Inverse Transform

| Panel | Content |
|-------|---------|
| **f(t) — Recovered Signal** | Time-domain signal with steady-state estimate line |
| **Pole-Zero Map** | Poles marked ×, zeros marked ○, stable left-half-plane shaded |
| **Partial Fractions** | Each PFD term shown |


## Examples

```
ℒ-calc » L  t**3
  Result F(s):  6 / (s^4)

ℒ-calc » L  exp(-t)*sin(2*t)
  Result F(s):  2 / (((s + 1)^2) + 4)

ℒ-calc » Li  1/(s*(s+1))
  Result f(t):  1 - (e^-t)

ℒ-calc » Li  s/(s**2 + 9)
  Result f(t):  cos(3t)
```

## Limitations

- Limited by Sympy's abilities.  Cannot perform transforms on piecewise functions.
- The 3D s-plane surface plot is restricted to Re(s) > 0 to avoid poles; functions with poles on the imaginary axis may show gaps.
- Very high-order rational ffunctions may produce slow or incomplete partial fraction decompositions.

