# Prune and Regrow

Shrinking a small neural network with pruning, NEAT-style regrowth and knowledge distillation, with interactive 3D replays of every step.

## ▶ Live demos

Open them in a browser, press **Play**, and drag to rotate the network.

<table>
<tr>
<td width="50%" align="center">
<a href="https://aghyada1.github.io/prune-and-regrow/prune-and-regrow.html"><img src="screenshots/pruning-final.png" alt="3D pruning replay"></a><br>
<b><a href="https://aghyada1.github.io/prune-and-regrow/prune-and-regrow.html">Prune and Regrow →</a></b><br>
Four pruning methods take a 3,466-parameter network down to ~160, round by round.
</td>
<td width="50%" align="center">
<a href="https://aghyada1.github.io/prune-and-regrow/distill/best-student.html"><img src="screenshots/student-final.png" alt="3D distilled NEAT student"></a><br>
<b><a href="https://aghyada1.github.io/prune-and-regrow/distill/best-student.html">Distilled NEAT student →</a></b><br>
NEAT evolves a 148-parameter student (93.7%) over 40 generations.
</td>
</tr>
</table>

## What this is

A 64-32-32-10 MLP trained on scikit-learn's 8x8 digits (3,466 parameters, 97.7% test accuracy) is cut down in six rounds to about 160 parameters. Each method gets the same fine-tuning budget, and every number is the mean of 3 seeds. The test set is 540 images, so one image is 0.19 points and differences under about a point are noise.

It is a small, CPU-only study meant to make pruning visible and to test one idea honestly, not a benchmark.

## Methods

| method | idea |
|---|---|
| Delete small weights | classic magnitude pruning, then fine-tune |
| Delete weak neurons | remove neurons with the smallest mean-ablation effect on the output |
| NEAT-merge (`neat_merge.py`) | merge several neurons into one NEAT-grown neuron, delete neurons, or trim links; pick the change that loses least per parameter saved, after re-fitting the surviving outgoing weights by least squares |
| NEAT-deep (`neat_deep.py`) | NEAT-merge on a general graph, so new neurons can open new hidden layers and links can skip layers |
| Distillation (`distill/`) | NEAT builds a student under a hard parameter cap, trained on the original network's soft labels |

## Results

Pruning (mean test accuracy over 3 seeds, from `summary.json`):

| method | ~460 params | ~285 params | ~160 params |
|---|---|---|---|
| Delete small weights | 85.6 | 77.8 | 49.0 |
| NEAT-merge, no new neurons (ablation) | 95.5 | 94.1 | 87.5 |
| **NEAT-merge** | **96.5** | **95.1** | **91.0** |
| NEAT-deep (new layers) | 95.7 | 94.8 | 90.7 |

Deleting weak neurons cannot get this small: it stops at 717 parameters and 93.6%.

Distillation (from `distill/README.md`):

| student | params | acc % |
|---|---|---|
| original network | 3466 | 97.7 |
| NEAT-merge smallest net | 163 | 91.0 |
| **NEAT from that net, distilled, cap 150** | **149** | **93.3** |
| same 163-param net, just trained longer | 163 | 93.4 |
| 163-param net with weakest links cut to 120 (no NEAT) | 119 | 83.6 |
| NEAT from that net, distilled, cap 120 | 120 | 91.1 |

## What I learned

- **Re-fitting is what makes NEAT-merge work.** After each change, the next layer's weights are re-solved so it still gets nearly the same signal. Without new neurons the same search is just as good down to about 290 parameters; growing new neurons pays off only at the smallest size (91.0% vs 87.5%).
- **New layers did not help.** NEAT-deep uses them (the recorded run ends with 5 hidden layers) but lands within seed noise of the flat version. On this task the limit seems to be how many pixel links survive, not depth.
- **Most of the distillation gain was longer training.** The 163-parameter net reaches 93.4% just by training longer. What NEAT plus distillation adds is the same accuracy with fewer parameters, and 91.1% at 120 parameters where plain link cutting drops to 83.6%.
- **Tiny fixed networks train badly from scratch** (36 to 75% at this size). NEAT avoids this because each child keeps its parent's trained weights.

## Run it

```bash
pip install -r requirements.txt
python3 compare.py                 # classic pruning vs first NEAT version, ~10 min
python3 neat_merge.py full3_look   # NEAT-merge winner, 3 seeds
python3 neat_deep.py deep          # NEAT-deep
cd distill && python3 distill.py 0 1 2 && python3 summarize.py   # ~45 min per seed
```

`demo.py` is the first toy experiment (needs matplotlib for `heatmap.png`). Per-seed partial outputs are in `runs/`. The 3D pages are self-contained HTML; they load three.js from a CDN.

## Layout

```
compare.py, neat_merge.py, neat_deep.py   pruning experiments (each imports the one before)
*.json, *.log                             their results
prune-and-regrow.html                     3D replay and write-up of the pruning experiments
distill/                                  NEAT + distillation, its own README and 3D replay
runs/                                     per-seed partial outputs
screenshots/                              stills from the 3D pages
```

## License

MIT
