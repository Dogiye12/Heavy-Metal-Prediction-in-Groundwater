🌊 Heavy-Metal-Prediction-in-Groundwater

A synthetic data–driven project to predict heavy metal concentrations (Lead & Arsenic) in groundwater using machine learning. The project simulates hydrochemical and environmental variables, generates datasets, and applies a Random Forest regression model for prediction and analysis.

📖 Overview

Groundwater contamination by heavy metals poses serious risks to public health. This project demonstrates:

Synthetic generation of groundwater quality data (>100 samples).

Feature engineering (pH, EC, TDS, hardness, depth, land use, etc.).

Prediction of Pb_ppb and As_ppb concentrations.

Model evaluation (R², RMSE, MAE).

Export of datasets, trained models, and evaluation artifacts.

The approach is designed for research, teaching, and demonstration purposes.

🚀 Features

Synthetic Dataset Generator: Produces realistic hydrochemical data.

Machine Learning Pipeline: Preprocessing + Random Forest regression.

Multiple Targets: Predict Lead (Pb_ppb) or Arsenic (As_ppb).

WHO Exceedance Labels: Binary flags for >10 ppb thresholds.

Artifacts Exported:

Excel/CSV dataset

Trained model (.joblib)

Metrics report

🛠️ Installation

Clone the repo and install requirements:

git clone https://github.com/your-username/Heavy-Metal-Prediction-in-Groundwater.git
cd Heavy-Metal-Prediction-in-Groundwater
pip install -r requirements.txt

📂 Project Structure
Heavy-Metal-Prediction-in-Groundwater/
│── heavy_metal_prediction.py   # Main script (data + modeling)
│── heavy_metal_synthetic_dataset.xlsx   # Example dataset
│── artifacts/                  # Auto-generated models, CSVs, plots
│── README.md                   # Documentation

▶️ Usage
Generate Dataset & Train Model
python heavy_metal_prediction.py --n 1000 --target Pb_ppb --seed 42


Arguments:

--n : Number of samples (≥101)

--target : Target metal (Pb_ppb or As_ppb)

--seed : Random seed (default=42)

Example Output
Rows: 1000   Target: Pb_ppb
Metrics: {'cv_mean_r2': 0.87, 'r2': 0.85, 'rmse': 5.3, 'mae': 4.1}

📊 Example Dataset Fields

Hydrochemical: pH, EC (µS/cm), TDS (mg/L), Hardness, Nitrate, Sulfate, Chloride, DO

Environmental: Depth, Distance to industry, Temperature, Rainfall, Land use

Targets: Pb_ppb, As_ppb

WHO Flags: Pb_exceeds_WHO_10ppb, As_exceeds_WHO_10ppb

🔒 Notes

All data is synthetic; not suitable for direct policy or management use.

For real-world studies, replace the synthetic generator with measured field/lab data.

🧩 Roadmap

 Add classification mode (exceedance prediction).

 Include geospatial coordinates for mapping.

 Implement SHAP feature attribution plots.

 Deploy via Streamlit app for interactive demo.

🤝 Contributing

Contributions are welcome! Fork the repo, improve the generator, or extend the ML pipeline, then submit a PR.

Author Name: Amos Meremu Dogiye
Github: https://github.com/Dogiye12
LinkedIn: https://www.linkedin.com/in/meremu-amos-993333314/



📜 License

This project is licensed under the MIT License.
