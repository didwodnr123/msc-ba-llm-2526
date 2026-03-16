# Evaluation Summary

Full per-label metrics for all models and prompting modes, evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw test set.

---

## Overall Metrics

| Model | Mode | Exact Match | Micro P | Micro R | Micro F1 | Macro F1 |
|-------|------|:-----------:|:-------:|:-------:|:--------:|:--------:|
| `gpt-4.1` | Zero-Shot *(baseline)* | 0.480 | 0.639 | 0.522 | 0.575 | 0.539 |
| `gpt-4.1` | Few-Shot-5 | 0.564 | 0.701 | 0.749 | 0.724 | 0.616 |
| `gpt-4.1` | Few-Shot-10 | 0.557 | 0.694 | 0.731 | 0.712 | 0.620 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.575 | 0.688 | 0.772 | **0.728** | 0.628 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.562 | 0.667 | 0.799 | 0.727 | 0.622 |
| `gpt-5-mini` | Zero-Shot | 0.504 | 0.704 | 0.041 | 0.077 | 0.043 |
| `gpt-5-mini` | Few-Shot-5 | 0.523 | 0.747 | 0.156 | 0.259 | 0.153 |
| `gpt-5-mini` | Few-Shot-10 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.501 | 0.647 | 0.044 | 0.083 | 0.044 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.568 | 0.719 | 0.715 | 0.717 | 0.604 |
| `gpt-5.4` | Few-Shot-5 | 0.592 | 0.745 | 0.760 | **0.753** | 0.608 |
| `gpt-5.4` | Few-Shot-10 | 0.556 | 0.741 | 0.680 | 0.710 | 0.619 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.589 | 0.734 | 0.730 | 0.732 | 0.635 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.595 | 0.725 | 0.759 | 0.741 | 0.644 |
| `toxic-bert` | Detoxify | 0.682 | 0.799 | 0.849 | 0.823 | 0.709 |
| `unbiased-toxic-roberta` | Detoxify | 0.692 | 0.803 | 0.868 | **0.834** | 0.708 |

---

## Per-Label F1 Scores

| Model | Mode | toxic | severe\_toxic | obscene | threat | insult | identity\_hate |
|-------|------|:-----:|:------------:|:-------:|:------:|:------:|:--------------:|
| `gpt-4.1` | Zero-Shot | 0.253 | 0.380 | 0.750 | 0.507 | 0.721 | 0.625 |
| `gpt-4.1` | Few-Shot-5 | 0.803 | 0.342 | 0.704 | 0.526 | 0.721 | 0.599 |
| `gpt-4.1` | Few-Shot-10 | 0.779 | 0.379 | 0.699 | 0.535 | 0.729 | 0.601 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.808 | 0.378 | 0.736 | 0.549 | 0.720 | 0.577 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.828 | 0.335 | 0.748 | 0.540 | 0.715 | 0.566 |
| `gpt-5-mini` | Zero-Shot | 0.091 | 0.000 | 0.056 | 0.000 | 0.102 | 0.007 |
| `gpt-5-mini` | Few-Shot-5 | 0.293 | 0.019 | 0.194 | 0.036 | 0.326 | 0.049 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.110 | 0.000 | 0.017 | 0.000 | 0.124 | 0.010 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.727 | 0.249 | 0.760 | 0.475 | 0.727 | 0.685 |
| `gpt-5.4` | Few-Shot-5 | 0.837 | 0.197 | 0.731 | 0.481 | 0.718 | 0.684 |
| `gpt-5.4` | Few-Shot-10 | 0.757 | 0.392 | 0.700 | 0.461 | 0.706 | 0.699 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.793 | 0.421 | 0.722 | 0.483 | 0.727 | 0.665 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.802 | 0.441 | 0.754 | 0.467 | 0.727 | 0.672 |
| `toxic-bert` | Detoxify | 0.906 | **0.451** | **0.819** | **0.593** | 0.774 | 0.709 |
| `unbiased-toxic-roberta` | Detoxify | **0.919** | 0.435 | 0.817 | 0.570 | **0.780** | **0.726** |

---

## Per-Label Precision & Recall

### toxic

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.925 | 0.146 | 0.253 |
| `gpt-4.1` | Few-Shot-5 | 0.858 | 0.755 | 0.803 |
| `gpt-4.1` | Few-Shot-10 | 0.883 | 0.697 | 0.779 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.864 | 0.759 | 0.808 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.862 | 0.796 | 0.828 |
| `gpt-5-mini` | Zero-Shot | 0.825 | 0.048 | 0.091 |
| `gpt-5-mini` | Few-Shot-5 | 0.853 | 0.177 | 0.293 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.763 | 0.059 | 0.110 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.912 | 0.604 | 0.727 |
| `gpt-5.4` | Few-Shot-5 | 0.870 | 0.807 | 0.837 |
| `gpt-5.4` | Few-Shot-10 | 0.881 | 0.664 | 0.757 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.875 | 0.725 | 0.793 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.871 | 0.742 | 0.802 |
| `toxic-bert` | Detoxify | 0.903 | 0.909 | 0.906 |
| `unbiased-toxic-roberta` | Detoxify | 0.882 | 0.959 | **0.919** |

### severe\_toxic

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.279 | 0.595 | 0.380 |
| `gpt-4.1` | Few-Shot-5 | 0.309 | 0.382 | 0.342 |
| `gpt-4.1` | Few-Shot-10 | 0.277 | 0.598 | 0.379 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.269 | 0.637 | 0.378 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.212 | 0.794 | 0.335 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 1.000 | 0.010 | 0.019 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.490 | 0.167 | 0.249 |
| `gpt-5.4` | Few-Shot-5 | 0.600 | 0.118 | 0.197 |
| `gpt-5.4` | Few-Shot-10 | 0.462 | 0.340 | 0.392 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.377 | 0.477 | 0.421 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.340 | 0.627 | 0.441 |
| `toxic-bert` | Detoxify | 0.385 | 0.546 | **0.451** |
| `unbiased-toxic-roberta` | Detoxify | 0.516 | 0.376 | 0.435 |

### obscene

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.798 | 0.707 | 0.750 |
| `gpt-4.1` | Few-Shot-5 | 0.818 | 0.619 | 0.704 |
| `gpt-4.1` | Few-Shot-10 | 0.801 | 0.620 | 0.699 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.782 | 0.695 | 0.736 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.792 | 0.708 | 0.748 |
| `gpt-5-mini` | Zero-Shot | 0.802 | 0.029 | 0.056 |
| `gpt-5-mini` | Few-Shot-5 | 0.849 | 0.110 | 0.194 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.867 | 0.009 | 0.017 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.763 | 0.757 | 0.760 |
| `gpt-5.4` | Few-Shot-5 | 0.778 | 0.690 | 0.731 |
| `gpt-5.4` | Few-Shot-10 | 0.783 | 0.632 | 0.700 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.796 | 0.661 | 0.722 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.776 | 0.734 | 0.754 |
| `toxic-bert` | Detoxify | 0.774 | 0.870 | **0.819** |
| `unbiased-toxic-roberta` | Detoxify | 0.801 | 0.833 | 0.817 |

### threat

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.357 | 0.877 | 0.507 |
| `gpt-4.1` | Few-Shot-5 | 0.381 | 0.847 | 0.526 |
| `gpt-4.1` | Few-Shot-10 | 0.391 | 0.847 | 0.535 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.417 | 0.804 | 0.549 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.410 | 0.791 | 0.540 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.750 | 0.018 | 0.036 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.321 | 0.914 | 0.475 |
| `gpt-5.4` | Few-Shot-5 | 0.332 | 0.877 | 0.481 |
| `gpt-5.4` | Few-Shot-10 | 0.316 | 0.853 | 0.461 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.336 | 0.859 | 0.483 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.318 | 0.877 | 0.467 |
| `toxic-bert` | Detoxify | 0.468 | 0.810 | **0.593** |
| `unbiased-toxic-roberta` | Detoxify | 0.563 | 0.577 | 0.570 |

### insult

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.613 | 0.875 | 0.721 |
| `gpt-4.1` | Few-Shot-5 | 0.614 | 0.873 | 0.721 |
| `gpt-4.1` | Few-Shot-10 | 0.626 | 0.872 | 0.729 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.623 | 0.853 | 0.720 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.609 | 0.866 | 0.715 |
| `gpt-5-mini` | Zero-Shot | 0.550 | 0.056 | 0.102 |
| `gpt-5-mini` | Few-Shot-5 | 0.612 | 0.222 | 0.326 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.516 | 0.071 | 0.124 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.617 | 0.884 | 0.727 |
| `gpt-5.4` | Few-Shot-5 | 0.653 | 0.798 | 0.718 |
| `gpt-5.4` | Few-Shot-10 | 0.662 | 0.755 | 0.706 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.650 | 0.825 | 0.727 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.660 | 0.808 | 0.727 |
| `toxic-bert` | Detoxify | 0.764 | 0.785 | 0.774 |
| `unbiased-toxic-roberta` | Detoxify | 0.722 | 0.848 | **0.780** |

### identity\_hate

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.474 | 0.917 | 0.625 |
| `gpt-4.1` | Few-Shot-5 | 0.439 | 0.940 | 0.599 |
| `gpt-4.1` | Few-Shot-10 | 0.438 | 0.959 | 0.601 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.412 | 0.962 | 0.577 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.400 | 0.967 | 0.566 |
| `gpt-5-mini` | Zero-Shot | 0.400 | 0.003 | 0.007 |
| `gpt-5-mini` | Few-Shot-5 | 0.441 | 0.026 | 0.049 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.600 | 0.005 | 0.010 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.566 | 0.867 | 0.685 |
| `gpt-5.4` | Few-Shot-5 | 0.569 | 0.857 | 0.684 |
| `gpt-5.4` | Few-Shot-10 | 0.601 | 0.836 | 0.699 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.580 | 0.778 | 0.665 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.565 | 0.828 | 0.672 |
| `toxic-bert` | Detoxify | 0.700 | 0.719 | 0.709 |
| `unbiased-toxic-roberta` | Detoxify | 0.732 | 0.721 | **0.726** |
