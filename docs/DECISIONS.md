# Decision Log

## D001 — Use yfinance as default source

### Decision
Use yfinance as the default public data source.

### Rationale
Stooq through pandas-datareader caused parsing failures in Codespaces. yfinance provides easier reproducibility for a public educational GitHub project.

### Consequence
The repo should explicitly state that public data are used for demonstration and research purposes only.

---

## D002 — Do not present raw 3-regime HMM as final

### Decision
The raw 3-regime HMM is treated as a diagnostic baseline, not as the final regime model.

### Rationale
Two regimes alternate daily, producing economically implausible one-day states.

### Consequence
The final version should use either a two-regime model, smoothed posterior probabilities, or minimum-duration filtering.

---

## D003 — Make SMDP stress states the project differentiator

### Decision
The SMDP-inspired stress-state layer should become the central contribution.

### Rationale
It connects directly to the author's research background in stochastic processes, semi-Markov decision models, degradation, and control.

### Consequence
The README and final notebook should emphasize stress accumulation and dynamic risk-intensity control.
