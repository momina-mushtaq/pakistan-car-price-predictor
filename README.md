Pakistan Used Car Price Predictor

Predicting used car prices in Pakistan from 77k+ listings using Python and scikit-learn, with a Streamlit app planned for live predictions.

Status: Data cleaning and exploratory analysis complete. Modeling and app in progress.

Dataset

Pakistan Used Car Prices 2023 (Kaggle): 77,237 used car listings with make, model, city, engine size, mileage, fuel type, transmission, assembly, body type, colour, registration city and price (PKR).

The data is not stored in this repository. To reproduce the analysis, download the CSV from Kaggle and save it as data/pakwheels_used_cars.csv.

Data quality notes
The year column is unusable. Every non-missing value is the constant 1905, and the raw text of the file shows the same, so the problem is in the source data. The column was excluded. Mileage is the closest available proxy for vehicle age, which limits how well the model can capture depreciation.
assembly is 69% missing (53,261 of 77,237 rows). Missing values were kept as an "Unknown" category instead of dropping the column.
Mileage contains a placeholder value. 123,456 km appears 945 times, far more often than any neighbouring value, so those rows were removed.
Data cleaning
Step	Rule	Rows removed	Rows left
1	Drop rows missing price or engine_cc	651	76,586
2	Remove placeholder mileage (123,456 km)	945	75,641
3	Keep mileage between 100 and 400,000 km	3,082	72,559
4	Keep engine size between 600 and 6,000 cc	189	72,370
5	Keep prices between the 1st and 99th percentile (PKR 330,000 to 20,531,000)	1,400	70,970

The cleaned dataset keeps 92.7% of the usable rows (70,970 of 76,586). Step 5 also removes some genuine very cheap and luxury listings. This is a deliberate trade-off that keeps extreme values from dominating the model, and it means the model is not suited to predicting prices outside that range.

Remaining missing values in body, color and fuel_type were filled with "Unknown".

Key findings
Prices are right-skewed. Median price is PKR 2.7M against a mean of about PKR 3.45M. I plan to model log(price).
Three brands dominate. Toyota, Suzuki and Honda make up about 82% of the raw listings, so the model will be most reliable for them.
Listings cluster in big cities. Lahore, Karachi and Islamabad together account for roughly half of the cleaned listings.
Most cars run on petrol: about 92% of listings with a fuel type, with diesel and hybrid making up the rest.
Transmission matters. The average price of automatic cars (about PKR 4.8M) is well over double that of manual cars (about PKR 1.8M).
Engine size is the strongest simple signal. Its correlation with price is 0.58.
Higher mileage goes with lower prices (correlation -0.30). This partly reflects vehicle age, which is not available because of the year issue.
Project structure
pakistan-car-price-predictor/
├── notebooks/
│   └── 01_eda.ipynb        # cleaning and exploratory analysis
├── data/                   # not tracked; place the Kaggle CSV here
├── .gitignore
└── README.md

Planned: notebooks/02_modeling.ipynb, app/app.py, requirements.txt.

Roadmap
 Data quality checks and cleaning
 Exploratory data analysis
 Baseline model (Linear Regression on log price)
 Random Forest and Gradient Boosting with cross-validation
 Feature importance and error analysis
 Streamlit app and deployment
 Final results table and limitations
Tools

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter
