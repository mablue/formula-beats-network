# Experiment Code

This directory contains the 33 experiments summarized in the paper.
Each labN.py is a self-contained Python script (requires only NumPy,
except figures which use Matplotlib).

## Quick start

    python code/lab33.py
    cat lab33_result.txt

## Experiment index

1.  lab_pure.py   - Binary quantization of transformer weights
2.  lab2.py       - Bit-sweep and layer sensitivity
3.  lab3.py       - Where is the signal
4.  lab4.py       - Replace layer with random or identity
5.  lab5.py       - Low-rank approximation
6.  lab6.py       - Weight sharing across layers
7.  lab7.py       - Weight sharing validation (falsified)
8.  lab8.py       - Neuron importance and sparsity
9.  lab9.py       - Structural sparsity via pruning
10. lab10.py      - Reservoir: fixed random middle layers
11. lab11.py      - Field emitters (failed)
12. lab12.py      - Fuzzy TSK inference (42x better on smooth)
13. lab13.py      - Fuzzy scaling to 8D
14. lab14.py      - Fuzzy on language (failed)
15. lab15.py      - Fair language comparison
16. lab16.py      - Formula-first: Poly vs Fourier vs MLP
17. lab17.py      - Random Features on language
18. lab18b.py     - Multi-layer RF (no gain from stacking)
19. lab19.py      - SINDy symbolic regression
20. lab20.py      - Tensor decomposition of language
21. lab21.py      - SINDy on embeddings
22. lab22b.py     - Standard vs Linear attention
23. lab23.py      - RF vs Fuzzy vs MLP on language
24. lab24.py      - RF stacking sweep
25. lab25.py      - MLP capacity sweep
26. lab26.py      - Growth-based structure discovery
27. lab27.py      - Multi-formula discovery
28. lab28.py      - Does more data help
29. lab29.py      - Structured formulas
30. lab30.py      - Oracle bound (ceiling)
31. lab31.py      - LLM text generation with RF
32. lab32.py      - RF vs MLP (2x2 fair comparison)
33. lab33.py      - Scale test (V=164, CTX=3/5)

## Result files

Each labN_result.txt (or .json) contains the raw output from that
experiment. These are included for reproducibility.

## Environment

- Hardware: Android ARM64, 8 GB RAM (Termux)
- Python: 3.12
- NumPy: 2.5.4
- Matplotlib: for figures only
- Total runtime: ~4 hours wall-clock across all 33 experiments

## Reproducing

    pkg install python python-numpy
    pip install matplotlib
    python code/lab33.py

## Notes

Several experiments have known bugs in hand-written backpropagation
(e.g. lab22). Where this occurred, results were validated with
numerical gradients (lab22c.py). The main results (lab26, lab30,
lab33) use closed-form ridge regression and are fully reliable.
