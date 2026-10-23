# Showcase

`python -m showcase.build` regenerates, from the benchmark results and the leaderboards' public results:

- `data/official.csv`: every official value shown, with its source and the date it was read;
- `data/same_protocol.csv`: GIFT-Eval in forced univariate mode, computed by us (`benchmarks/competitors.yaml`);
- `figures/pareto.png`, `figures/same_protocol.png`, `figures/banner.png` (a real forecast of the model);
- `README.md` and `MODEL_CARD.md` from `README.template.md` and `MODEL_CARD.template.md`.

```bash
python -m showcase.build --gift-eval gift-eval --fev-bench fev/benchmarks/fev_bench \
    --time-output TIME-Output/results --time-naive TIME-Output/results --model weights
```

`TIME-Output` is `huggingface.co/datasets/Real-TSF/TIME-Output` at revision 4c93dea (only the `metrics.npz` files of the
models listed in `models.yaml` are needed).

## The figures show a selection

`models.yaml` lists well-known models and the models we ran ourselves; many leaderboard entries are left out of the
figures. Sizes come from the official model cards (the Hub's exact count when it exists).

## How the head sentence was checked

The dated claim of the head sentence ("at release, the most accurate on GIFT-Eval, fev-bench and TIME among models under
10M parameters with public weights and a permissive license") was checked on 2026-10-22 against **every** entry of the three leaderboards,
not only the models in the figures. A model ranked above ours on a benchmark is set aside only for one of these reasons:

| reason | examples |
|---|---|
| agent, ensemble or system of several models | most of the GIFT-Eval entries above ours (since 2026-10-22: PERSONA-AI-T4, RegLab-AH-2, ChakraTS-Lab, Granite-timeseries-ensemble-r1) |
| fine-tuned on the benchmark data, or declared test-data leakage | GIFT-Eval: Tabby, TurkForecast-FM-Chronos2-LoRA-v1 |
| no public weights (closed API, weights not released, or no model link) | QiYao-M, DeOSAlphaTimeGPTPredictor-2025, Falcon-2.0, Lingjiang2.0, EXAONE-Forecast, EFG-base, Migas-1.0, TimeRSI, ChakraTS, Morpheus-2.0, CHARM (63M, API only) |
| more than 10M parameters (official card), or built on such a model | STRIDE (+Chronos-2 / +Timer-S1), Chronos-2, TimesFM-2.5 / 3, Toto-2.0 1B / 2.5B, TiRex-2, t0-beta / gamma, Granite-PatchTST-FM-r2, tafsut, Zeus, Timer-S1, Timestral-28M / 123M / 209M, TimeH2O-v1 (109M), PCFM (52.9M) |
| non-commercial or restricted license | TimeH2O-v1-tiny (5.8M, fev-bench and GIFT-Eval), Timestral-5M (5.2M, fev-bench, TIME and GIFT-Eval CRPS) |

Once these are set aside, the only models under 10M parameters ranked above ours are TimeH2O-v1-tiny and Timestral-5M,
both under restricted licenses. The check should be repeated whenever the leaderboards change.

**Re-checked on 2026-10-23 on GIFT-Eval** (gift-eval `cfaefe0`, 158 result folders; the copy used before, `6fdb10d` of
2026-07-06, predated 18 entries, among them Timestral, merged on 2026-10-16). Every entry was rescored against the
same Seasonal Naive (identical in both copies). The new entries under 10M parameters below ours on both metrics:
TimeSSM-2.5M (0.7544 / 0.5128), Mobius-0 (0.7140 / 0.4959), Yug (0.7443 / 0.5287). EFG-base (0.7006 / 0.48159, no public
weights) is just behind ours (0.7003 / 0.48156). The sentence holds.

**fev-bench and TIME re-checked on 2026-10-23.** fev `ceb25ae` adds TimeH2O-v1 (109M parameters, non-commercial) and
TimeRSI-1.5 (closed API), and now hides TimeH2O-v1-tiny on the site; our skill scores are unchanged and our rank too
(7th in SQL, 6th in MASE, of 33 visible models). TIME: the results (`Real-TSF/TIME-Output` `4c93dea`) are the latest; the
evaluation code changed only by new model scripts. GIFT-Eval: the evaluation code (`src/`) and the task lists of the
official notebooks are identical in `6fdb10d` and `cfaefe0`. The sentence holds.
