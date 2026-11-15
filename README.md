# makefun-polymarket-faa-attribution-01-v2

Minute-by-minute Yes-price forecasts for Polymarket's FAA outage attribution
markets, 4 hours ahead, as five quantiles (5/25/50/75/95%) per market-minute.

Method: empirical quantiles, recentered on persistence. Prediction-market
prices are near-martingales, so the median forecast is the last observed
price. The spread is learned from each market's own last 8 hours:
empirical quantiles of the h-minute price changes at a coarse grid of
horizons, interpolated in log(h) to every horizon 1..240, then shifted so
the median sits on the last price. The interquartile range is shrunk
(x0.50) because prices mostly sit still, while the 5%/95% tails keep their
full jump width. Horizons with too little history fall back to the house
baseline spread (0.0025 per square-root minute). All quantiles are clamped
to [0, 1], rounded to 6 decimals, and forced non-decreasing.

Practice result: on the kit's six practice cases (p00-p05), scored against
Polymarket's public price history for the 4 hours after each checkpoint,
this entry scores a pinball loss of 0.287804, against the house baseline's
0.363589 (median error 0.767503, 90% coverage 0.949642). The 8-hour
lookback and 0.50 inner-band shrink were chosen by testing nearby settings
on the same practice cases (10-hour / 0.65 scored 0.293157; 6-hour / 0.50
scored 0.289124). Practice uses a different event, so its score is no
guide to the final.

Run: `MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> python3 main.py`.
Python 3.13, standard library only, deterministic (two runs are
byte-identical).
