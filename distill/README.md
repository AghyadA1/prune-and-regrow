# Distilling the original network into a NEAT-built student

Goal: a student **smaller than ~160 parameters** and **more accurate than 91.0%**, which is the smallest NEAT-merge result (`full3_look`, mean of 3 seeds).
Everything uses the same digits split and the same parameter count (active links + hidden biases + 10 output biases) as `../neat_merge.py`. The test set is 540 images, so one image is 0.19 points.

## Result (mean test accuracy over seeds 0, 1, 2)

| student | params | acc % | min–max |
|---|---|---|---|
| teacher (original 64-32-32-10) | 3466 | 97.7 | 97.6–98.0 |
| **bar:** NEAT-merge smallest net, as in the earlier thread | 163 | 91.0 | 88.9–92.4 |
| **NEAT from that net, distilled, cap 150** | **149** | **93.3** | 92.8–93.7 |
| NEAT from that net, distilled, cap 120 | 120 | 91.1 | 89.4–92.2 |
| NEAT from scratch, distilled, cap 150 | 123 | 91.7 | 91.3–92.0 |
| *controls* | | | |
| same 163-param net, just trained longer on labels | 163 | 93.4 | 91.9–94.3 |
| same 163-param net, trained longer with distillation | 163 | 93.2 | 92.6–94.1 |
| 163-param net with weakest links cut to 150, distilled (no NEAT) | 150 | 92.0 | 90.6–92.8 |
| same, cut to 120 (no NEAT) | 119 | 83.6 | 78.7–86.1 |
| NEAT from that net, labels only, cap 150 (no distillation) | 150 | 92.0 | 90.4–93.2 |
| NEAT from scratch, labels only, cap 150 | 108 | 90.4 | 87.4–92.0 |
| plain fixed 64-2-10 net, labels / distilled | 160 | 61.4 / 36.4 | |
| fixed random sparse net, same size as the NEAT-from-scratch winner, labels / distilled | 123 | 74.5 / 74.8 | |

## What it means

- **Target met.** NEAT started from the smallest network and distilled from the original reaches 93.3% at 149 parameters. Every seed beats its own 160-parameter starting point. NEAT from scratch reaches 91.7% at 123 parameters.
- **Most of the jump from 91.0% is longer training, not distillation.** The earlier 160-parameter networks were fine-tuned for only 20 epochs. Trained longer, the same network reaches 93.4%. What NEAT adds is reaching that accuracy with fewer parameters: 93.3% at 149 instead of 163, and 91.1% at 120 where cutting links without NEAT drops to 83.6%.
- **Distillation helps inside NEAT, not on a fixed net.** NEAT from the small network gets 93.3% distilled versus 92.0% on labels only, at the same 150 parameters. On the fixed 163-parameter network, distillation and labels are equal (93.2% vs 93.4%).
- **Fixed-size networks this small do not train well.** At this size a plain fixed network trained from scratch reaches 36–75%. Training a sparse network from random weights often leaves neurons dead. NEAT avoids this because each child keeps its parent's trained weights.

## How NEAT is pushed small and accurate (`distill.py`)

- Hard cap: a genome never exceeds `CAP` parameters. Any mutation that goes over is repaired by deleting its weakest links (|w| × activation std).
- Fitness: distillation loss on 20% of the training images held out from the search, plus 2e-4 per parameter, so the smaller of two equal students wins. The test set is used only for the final numbers.
- Lamarckian: each child trains for 300 Adam steps and keeps its weights, so NEAT only chooses the topology.
- Mutations: add link (70% gradient-guided, choosing the missing link with the largest gradient), add neuron (split a link), delete weakest links, delete neuron. Also NEAT crossover by innovation number and speciation.
- Distillation: KL to the teacher at T=2 plus 0.2×cross-entropy on real labels. The transfer set is the training images, 4 noisy copies and 8 mixups, all labelled by the teacher. Shifted copies were tried first and dropped: they cost a 160-parameter network about 20 points, because the test digits are centred.
- Settings: 30 genomes for 40 generations. The winner then trains for 6000 steps on all training images. Every network gets the same final training.

## Files

- `distill.py` runs the experiment. `python3 distill.py 0 1 2`; env `DS_CAP`, `DS_ONLY`, `DS_OUT`. About 45 min per seed on one CPU core.
- `exp_kd.py`: sanity check of distillation on the fixed 160-parameter network.
- `summarize.py` writes `summary.json` and prints the table.
- `run_seed*.json`: main runs at cap 150. `cap120_seed*.json`: cap 120. `hardseed_seed*.json`: the labels-only control. Each NEAT entry stores the winning genome and its per-generation history.
- `nm160_seed*.json`: the rebuilt 160-parameter NEAT-merge networks. They match the earlier thread exactly (91.67 / 92.41 / 88.89%).
- `best-student.html` is the 3D replay of the best run (seed 1): the starting network, the cut to 150 parameters, the best network of each of the 40 generations, and the final training. It is built by `record_evolution.py`, which re-runs the run deterministically and saves every generation to `evolution_seed1.json`, and by `export_evolution.py` with `evo_template.html`. The re-run ends on the same network as the original (148 parameters, 93.7%).
