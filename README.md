# Formula Beats Network

**Sparse Random Features Outperform Trained Neural Networks for Next-Token Prediction**

Masoud Azizi - University of Tabriz
- Email: mablue92@gmail.com
- Web: https://masoudsoft.ir
- ORCID: 0000-0002-8864-0533
- GitHub: https://github.com/mablue

## Results

| Model | Test Acc | Params | Train Time |
|---|---|---|---|
| MLP (h=512) | 38.3% | 32768 | 53.3s |
| **RF (D=1024)** | **43.2%** | 65536 | **2.72s** |

- +4.9 percentage points more accurate
- 20x faster training
- No backpropagation, no attention, no FFN

## Files

- paper.tex - LaTeX source
- paper.pdf - compiled paper
- figures/ - 5 figures
- gen_figs.py - figure generation

## License

MIT
