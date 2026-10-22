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

The head sentence ("the most accurate forecasting model under 10M parameters with public weights and a permissive
license, on GIFT-Eval, fev-bench and TIME") was checked on 2026-10-22 against **every** entry of the three leaderboards,
not only the models in the figures. A model ranked above ours on a benchmark is set aside only for one of these reasons:

| reason | examples |
|---|---|
| agent, ensemble or system of several models | most of the GIFT-Eval entries above ours |
| fine-tuned on the benchmark data, or declared test-data leakage | GIFT-Eval: Tabby, TurkForecast-FM-Chronos2-LoRA-v1 |
| no public weights (closed API, weights not released, or no model link) | QiYao-M, DeOSAlphaTimeGPTPredictor-2025, Falcon-2.0, Lingjiang2.0, EXAONE-Forecast, EFG-base, Migas-1.0, TimeRSI, ChakraTS |
| more than 10M parameters (official card), or built on such a model | STRIDE (+Chronos-2 / +Timer-S1), Chronos-2, TimesFM-2.5 / 3, Toto-2.0 1B / 2.5B, TiRex-2, t0-beta / gamma, Granite-PatchTST-FM-r2, tafsut, Zeus, Timer-S1, Timestral-28M / 123M / 209M |
| non-commercial or restricted license | TimeH2O-v1-tiny (5.8M, fev-bench), Timestral-5M (5.2M, fev-bench and TIME) |

Once these are set aside, the only models under 10M parameters ranked above ours are TimeH2O-v1-tiny and Timestral-5M,
both under restricted licenses. The check should be repeated whenever the leaderboards change.
