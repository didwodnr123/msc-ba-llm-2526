# Evaluation Summary

Full per-label metrics for all models and prompting modes, evaluated on **10,000 sampled comments** (5,000 toxic / 5,000 non-toxic) from the Jigsaw test set.

---

## Overall Metrics

| Model | Mode | Exact Match | Micro P | Micro R | Micro F1 | Macro F1 |
|-------|------|:-----------:|:-------:|:-------:|:--------:|:--------:|
| `gpt-4.1` | Zero-Shot *(baseline)* | 0.500 | 0.617 | 0.518 | 0.563 | 0.528 |
| `gpt-4.1` | Few-Shot-5 | 0.540 | 0.662 | 0.786 | 0.718 | 0.585 |
| `gpt-4.1` | Few-Shot-10 | 0.600 | 0.674 | 0.866 | 0.758 | 0.642 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.590 | 0.684 | 0.830 | 0.750 | 0.618 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.590 | 0.667 | 0.893 | 0.763 | 0.644 |
| `gpt-5-mini` | Zero-Shot | 0.500 | 0.500 | 0.018 | 0.034 | 0.013 |
| `gpt-5-mini` | Few-Shot-5 | 0.530 | 0.765 | 0.116 | 0.202 | 0.115 |
| `gpt-5-mini` | Few-Shot-10 | 0.500 | 0.571 | 0.036 | 0.067 | 0.032 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.510 | 0.643 | 0.080 | 0.143 | 0.108 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.530 | 0.900 | 0.080 | 0.148 | 0.138 |
| `gpt-5.4` | Zero-Shot | 0.610 | 0.735 | 0.768 | 0.751 | 0.592 |
| `gpt-5.4` | Few-Shot-5 | 0.600 | 0.707 | 0.839 | 0.767 | 0.587 |
| `gpt-5.4` | Few-Shot-10 | 0.610 | 0.769 | 0.741 | 0.755 | 0.673 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.630 | 0.718 | 0.839 | 0.774 | 0.633 |
| `gpt-5.4` | Few-Shot-10-Synth | **0.620** | 0.724 | 0.866 | **0.789** | 0.640 |
| `toxic-bert` | Detoxify | 0.810 | 0.857 | 0.911 | **0.883** | 0.776 |
| `unbiased-toxic-roberta` | Detoxify | 0.720 | 0.850 | 0.813 | 0.831 | 0.647 |

---

## Per-Label F1 Scores

| Model | Mode | toxic | severe\_toxic | obscene | threat | insult | identity\_hate |
|-------|------|:-----:|:------------:|:-------:|:------:|:------:|:--------------:|
| `gpt-4.1` | Zero-Shot | 0.286 | 0.500 | 0.766 | 0.250 | 0.667 | 0.700 |
| `gpt-4.1` | Few-Shot-5 | 0.867 | 0.308 | 0.735 | 0.286 | 0.646 | 0.667 |
| `gpt-4.1` | Few-Shot-10 | 0.891 | 0.556 | 0.808 | 0.222 | 0.677 | 0.700 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.891 | 0.400 | 0.808 | 0.286 | 0.656 | 0.667 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.907 | 0.556 | 0.830 | 0.222 | 0.646 | 0.700 |
| `gpt-5-mini` | Zero-Shot | 0.080 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.222 | 0.000 | 0.267 | 0.000 | 0.200 | 0.000 |
| `gpt-5-mini` | Few-Shot-10 | 0.118 | 0.000 | 0.000 | 0.000 | 0.074 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.185 | 0.000 | 0.074 | 0.000 | 0.138 | 0.250 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.154 | 0.000 | 0.077 | 0.000 | 0.154 | 0.444 |
| `gpt-5.4` | Zero-Shot | 0.819 | 0.000 | 0.840 | 0.444 | 0.710 | 0.737 |
| `gpt-5.4` | Few-Shot-5 | 0.887 | 0.000 | 0.824 | 0.400 | 0.677 | 0.737 |
| `gpt-5.4` | Few-Shot-10 | 0.810 | 0.500 | 0.766 | 0.444 | 0.741 | 0.778 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.875 | 0.250 | 0.846 | 0.400 | 0.690 | 0.737 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.905 | 0.250 | 0.836 | 0.400 | 0.712 | 0.737 |
| `toxic-bert` | Detoxify | **0.968** | 0.444 | **0.893** | **0.667** | **0.824** | **0.857** |
| `unbiased-toxic-roberta` | Detoxify | 0.957 | 0.286 | 0.889 | 0.667 | 0.720 | 0.364 |

---

## Per-Label Precision & Recall

### toxic

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 1.000 | 0.167 | 0.286 |
| `gpt-4.1` | Few-Shot-5 | 0.929 | 0.813 | 0.867 |
| `gpt-4.1` | Few-Shot-10 | 0.932 | 0.854 | 0.891 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.932 | 0.854 | 0.891 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.898 | 0.917 | 0.907 |
| `gpt-5-mini` | Zero-Shot | 1.000 | 0.042 | 0.080 |
| `gpt-5-mini` | Few-Shot-5 | 1.000 | 0.125 | 0.222 |
| `gpt-5-mini` | Few-Shot-10 | 1.000 | 0.063 | 0.118 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.833 | 0.104 | 0.185 |
| `gpt-5-mini` | Few-Shot-10-Synth | 1.000 | 0.083 | 0.154 |
| `gpt-5.4` | Zero-Shot | 0.971 | 0.708 | 0.819 |
| `gpt-5.4` | Few-Shot-5 | 0.878 | 0.896 | 0.887 |
| `gpt-5.4` | Few-Shot-10 | 0.944 | 0.708 | 0.810 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.875 | 0.875 | 0.875 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.915 | 0.896 | 0.905 |
| `toxic-bert` | Detoxify | 0.979 | 0.958 | 0.968 |
| `unbiased-toxic-roberta` | Detoxify | 0.978 | 0.938 | 0.957 |

### severe\_toxic

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.500 | 0.500 | 0.500 |
| `gpt-4.1` | Few-Shot-5 | 0.286 | 0.333 | 0.308 |
| `gpt-4.1` | Few-Shot-10 | 0.417 | 0.833 | 0.556 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.333 | 0.500 | 0.400 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.417 | 0.833 | 0.556 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Few-Shot-5 | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Few-Shot-10 | 1.000 | 0.333 | 0.500 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.500 | 0.167 | 0.250 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.500 | 0.167 | 0.250 |
| `toxic-bert` | Detoxify | 0.667 | 0.333 | 0.444 |
| `unbiased-toxic-roberta` | Detoxify | 1.000 | 0.167 | 0.286 |

### obscene

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.818 | 0.720 | 0.766 |
| `gpt-4.1` | Few-Shot-5 | 0.750 | 0.720 | 0.735 |
| `gpt-4.1` | Few-Shot-10 | 0.778 | 0.840 | 0.808 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.778 | 0.840 | 0.808 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.786 | 0.880 | 0.830 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.800 | 0.160 | 0.267 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.500 | 0.040 | 0.074 |
| `gpt-5-mini` | Few-Shot-10-Synth | 1.000 | 0.040 | 0.077 |
| `gpt-5.4` | Zero-Shot | 0.840 | 0.840 | 0.840 |
| `gpt-5.4` | Few-Shot-5 | 0.808 | 0.840 | 0.824 |
| `gpt-5.4` | Few-Shot-10 | 0.818 | 0.720 | 0.766 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.815 | 0.880 | 0.846 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.767 | 0.920 | 0.836 |
| `toxic-bert` | Detoxify | 0.806 | 1.000 | 0.893 |
| `unbiased-toxic-roberta` | Detoxify | 0.828 | 0.960 | 0.889 |

### threat

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.167 | 0.500 | 0.250 |
| `gpt-4.1` | Few-Shot-5 | 0.200 | 0.500 | 0.286 |
| `gpt-4.1` | Few-Shot-10 | 0.143 | 0.500 | 0.222 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.200 | 0.500 | 0.286 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.143 | 0.500 | 0.222 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10-Synth | 0.000 | 0.000 | 0.000 |
| `gpt-5.4` | Zero-Shot | 0.286 | 1.000 | 0.444 |
| `gpt-5.4` | Few-Shot-5 | 0.250 | 1.000 | 0.400 |
| `gpt-5.4` | Few-Shot-10 | 0.286 | 1.000 | 0.444 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.250 | 1.000 | 0.400 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.250 | 1.000 | 0.400 |
| `toxic-bert` | Detoxify | 0.500 | 1.000 | 0.667 |
| `unbiased-toxic-roberta` | Detoxify | 1.000 | 0.500 | 0.667 |

### insult

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.538 | 0.875 | 0.667 |
| `gpt-4.1` | Few-Shot-5 | 0.512 | 0.875 | 0.646 |
| `gpt-4.1` | Few-Shot-10 | 0.537 | 0.917 | 0.677 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.541 | 0.833 | 0.656 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.512 | 0.875 | 0.646 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.500 | 0.125 | 0.200 |
| `gpt-5-mini` | Few-Shot-10 | 0.333 | 0.042 | 0.074 |
| `gpt-5-mini` | Few-Shot-5-Synth | 0.400 | 0.083 | 0.138 |
| `gpt-5-mini` | Few-Shot-10-Synth | 1.000 | 0.083 | 0.154 |
| `gpt-5.4` | Zero-Shot | 0.579 | 0.917 | 0.710 |
| `gpt-5.4` | Few-Shot-5 | 0.553 | 0.875 | 0.677 |
| `gpt-5.4` | Few-Shot-10 | 0.667 | 0.833 | 0.741 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.588 | 0.833 | 0.690 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.600 | 0.875 | 0.712 |
| `toxic-bert` | Detoxify | 0.778 | 0.875 | 0.824 |
| `unbiased-toxic-roberta` | Detoxify | 0.692 | 0.750 | 0.720 |

### identity\_hate

| Model | Mode | Precision | Recall | F1 |
|-------|------|:---------:|:------:|:--:|
| `gpt-4.1` | Zero-Shot | 0.538 | 1.000 | 0.700 |
| `gpt-4.1` | Few-Shot-5 | 0.500 | 1.000 | 0.667 |
| `gpt-4.1` | Few-Shot-10 | 0.538 | 1.000 | 0.700 |
| `gpt-4.1` | Few-Shot-5-Synth | 0.500 | 1.000 | 0.667 |
| `gpt-4.1` | Few-Shot-10-Synth | 0.538 | 1.000 | 0.700 |
| `gpt-5-mini` | Zero-Shot | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-10 | 0.000 | 0.000 | 0.000 |
| `gpt-5-mini` | Few-Shot-5-Synth | 1.000 | 0.143 | 0.250 |
| `gpt-5-mini` | Few-Shot-10-Synth | 1.000 | 0.286 | 0.444 |
| `gpt-5.4` | Zero-Shot | 0.583 | 1.000 | 0.737 |
| `gpt-5.4` | Few-Shot-5 | 0.583 | 1.000 | 0.737 |
| `gpt-5.4` | Few-Shot-10 | 0.636 | 1.000 | 0.778 |
| `gpt-5.4` | Few-Shot-5-Synth | 0.583 | 1.000 | 0.737 |
| `gpt-5.4` | Few-Shot-10-Synth | 0.583 | 1.000 | 0.737 |
| `toxic-bert` | Detoxify | 0.857 | 0.857 | 0.857 |
| `unbiased-toxic-roberta` | Detoxify | 0.500 | 0.286 | 0.364 |
