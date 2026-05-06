# NBA MVP Predictor

Predicting the NBA Most Valuable Player from per-season player statistics.
Trained on 1982-2022 data; the 2023-24 prediction step scrapes
[basketball-reference.com](https://www.basketball-reference.com) for current
stats. This is a 2024 snapshot, not an actively retrained model.

Two framings are compared:

- **Regression baseline** (Decision Tree, Random Forest) — predict each
  player's MVP vote share, take the season's top-ranked player as the MVP.
- **CNN classifier** — different framing: each example is a `(10 players,
  N features)` matrix sampled from the season's top vote-getters; the
  network outputs a softmax over the 10 positions and is trained to pick
  the actual MVP from the pool.

Both use the same eligibility filter (cutoffs on VORP, games, minutes, USG%,
PER, WS) so candidate sets line up across models.

## Held-out seasons

The Random Forest is evaluated on five seasons left out of training. Top-ranked player by predicted vote share vs. the actual winner:

| Season | Predicted MVP        | Actual MVP            | Match |
|--------|----------------------|------------------------|:-----:|
| 1985   | Larry Bird           | Larry Bird             |   ✓   |
| 2009   | LeBron James         | LeBron James           |   ✓   |
| 2011   | LeBron James         | Derrick Rose           |   ✗   |
| 2019   | Giannis Antetokounmpo| Giannis Antetokounmpo  |   ✓   |
| 2022   | Nikola Jokić         | Nikola Jokić           |   ✓   |

4/5 on the holdout. The 2011 miss is the upset season where Derrick Rose won
over LeBron, which most contemporary models also got wrong. Decision Tree
gets 3/5. The CNN gets ~45% per-partition accuracy across the shuffled
10-player groups, which is well above the 10% chance baseline; its softmax
output is not calibrated to predict vote share. Run-by-run reports are in
`output/`.

## Install and run

Python 3.11 recommended (TensorFlow 2.x doesn't ship wheels for newer Python yet).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` runs end-to-end: train all three models, evaluate on the holdout
seasons (writes `output/report_*.txt`), then scrape `basketball-reference.com`
for current-season stats and print the 2023-24 MVP leaderboard.

The live-prediction step depends on basketball-reference's HTML structure.
Holdout evaluation runs and writes its reports before scraping, so even if
the scrape fails the model evaluation results are produced.

## Layout

```
Data/                       Training CSV and team-abbreviation lookup
src/PreProcessing/          Eligibility filter, KNN imputation, encoding
src/modeling/               Three model trainers + evaluation utilities
src/get_data_online/        basketball-reference.com scraper
cleaning_dataset.ipynb      EDA: choosing the eligibility cutoffs
modeling.ipynb              Grid search and model comparison
main.py                     End-to-end pipeline
```

## Data

Training data: [NBA Player Season Statistics with MVP Win Share](https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share)
on Kaggle, originally scraped from basketball-reference.com. Committed to the
repo at `Data/NBA_Dataset.csv` for reproducibility.

## License

MIT. See `LICENSE`.
