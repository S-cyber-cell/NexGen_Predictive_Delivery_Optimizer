# NexGen Predictive Delivery Optimizer (Streamlit)

A practical prototype to **predict late deliveries** and flag **at-risk orders** early, so operations can act before SLAs are breached.

## Folder layout
```
NexGen_Predictive_Delivery_Optimizer/
├─ app.py
├─ requirements.txt
├─ README.md
├─ Innovation_Brief.md
└─ data/
   ├─ orders.csv
   ├─ delivery_performance.csv
   ├─ routes_distance.csv
   ├─ vehicle_fleet.csv
   ├─ warehouse_inventory.csv
   ├─ customer_feedback.csv
   └─ cost_breakdown.csv
```

> Place the **7 CSV files** inside the `data/` folder with the exact filenames shown above.

## Quick start
1. Create a virtual environment (recommended)
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app
   ```bash
   streamlit run app.py
   ```
4. In the app, choose:
   - **Use CSVs from ./data** (default), or
   - Upload files via the sidebar

## What the app does
- Loads and merges the datasets on `order_id` (best-effort, resilient to missing columns)
- Builds a feature table with derived fields:
  - `is_late` (label) from `actual_delivery_hours` vs `promised_delivery_hours`
  - Traffic and weather flags
  - Priority encoding
- **EDA**: basic charts with matplotlib (priority mix, lateness histogram, distance vs lateness, top lanes)
- **Model**: RandomForest with tunable parameters and threshold
- **At-Risk Orders**: predicts late risk for in-transit orders and estimates business impact
- **Downloads**: export feature table and predicted at-risk orders

## Notes
- Works even if some files lack certain columns (they're filled with `NaN` or zeros).
- Model quality depends on label coverage and data quality.
- Keep it small and readable. This is meant to look like a real intern submission.