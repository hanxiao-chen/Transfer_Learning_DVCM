# Seoul Bike Sharing Demand Experiment

- `Y`: `log1p(Rented Bike Count)`
- `U`: hour of day
- `X`: weather variables plus season and holiday indicators
- Bandwidth: selected by target-validation CV for TL
- Figure: `figures/seoul_bike_log_rentals_hour_mse_with_y_profile_bandwidth_cv.pdf`

Run:

```bash
python seoul_bike/seoul_bike_experiment.py
python seoul_bike/plot_seoul_bike.py
```
