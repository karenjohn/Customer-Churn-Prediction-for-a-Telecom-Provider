
# Telecom Customer Churn Prediction

A machine learning project that predicts which customers of an anonymous telecom provider ("Company A") are likely to leave within 31 to 60 days, and turns the results into a practical retention proposal.

Built for the **GCI World 2026 April** final assignment (Matsuo Lab).

---

## Project Summary

Company A holds records on about 100,000 customers but has no in house machine learning capability. The goal of this project is to:

* Explore the data to understand what separates customers who leave from those who stay
* Build and compare machine learning models that flag at risk customers early
* Use the findings to propose a tiered retention strategy with an estimated business impact

The final proposal is written for business readers, so every technical result is linked back to an action the company could take.

---

## Business Problem

Telecom is a mature, crowded market, and keeping an existing customer costs far less than winning a new one. Major carriers compete on price and network quality, and churn is usually triggered by device upgrades, contract expiry, and competitor offers. If Company A can identify likely leavers 31 to 60 days ahead, it can contact them with a targeted offer before they go.

**Target variable:** `churn` (1 = customer left within 31 to 60 days, 0 = customer stayed)

---

## Repository Contents

```
.
├── README.md
├── churn_prediction.ipynb     # Full analysis notebook (rename to match your file)
└── excel files.zip/
    ├── Client.csv             # Customer information
    └── Record.csv             # Usage history
```

---

## Dataset

The data was provided as part of the GCI World course and is anonymised. It comes as two CSV files that are joined on `Customer_ID`.

| File | Rows | Columns | Content |
| --- | --- | --- | --- |
| `Client.csv` | 100,000 | 50 | Customer and household information such as handset, tenure, credit class, and demographics |
| `Record.csv` | 100,000 | 51 | Usage and billing history such as minutes of use, revenue, dropped calls, and the `churn` label |

After merging, the combined table has 100,000 rows and 100 columns.

**Things to know about the data**

* The classes are almost perfectly balanced (about 49.6% churned, 50.4% stayed). This is a property of the sample, not of the real customer base, so the churn rate here should not be read as Company A's true churn rate.
* Several columns have missing values, in some cases for nearly half of the customers (for example, number of cars and dwelling size).
* Some columns have limited documentation. The client did not share further detail because of data protection policies, so the analysis works within those constraints.

---

## How to Run the Project

### Option 1: Google Colab with Google Drive (how the notebook was originally built)

1. Create a folder named **Final Assignment** in your Google Drive (inside My Drive).
2. Upload `Client.csv` and `Record.csv` into that folder.
3. Upload the notebook to Colab, or open it from GitHub using **File > Open notebook > GitHub**.
4. Run the cells from top to bottom. When the Drive cell runs, approve the access request.

The notebook expects the data at `/content/drive/MyDrive/Final Assignment`.

### Option 2: Run locally or in Colab without Google Drive

1. Clone this repository and keep the CSV files inside the `data` folder.
2. Install the requirements:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

3. In the notebook, skip the Google Drive cell and load the data like this instead:

```python
client = pd.read_csv('data/Client.csv')
record = pd.read_csv('data/Record.csv')
```

4. Run the notebook from top to bottom.

### Requirements

* Python 3.9 or newer
* pandas, numpy, matplotlib, seaborn
* scikit-learn
* xgboost

---

## What Each Section of the Notebook Does

The notebook follows the same order as a typical consulting analysis, and each section builds on the one before it. Run the sections in order.

| Section | What it does |
| --- | --- |
| **1. Introduction and Market Analysis** | Sets out the telecom market context, cited industry figures on churn, and the business objective. |
| **2. Setup and Data Loading** | Imports the libraries, loads `Client.csv` and `Record.csv`, and merges them on `Customer_ID` into one table of 100,000 rows. |
| **3. Exploratory Data Analysis** | Checks missing values and the churn split, then compares churn rates across equipment age, change in monthly usage, and tenure. The charts show where churn is highest. |
| **4. Problem Definition** | States the task as binary classification and explains why AUC ROC is the main metric, with accuracy and F1 score as secondary metrics. |
| **5. Feature Engineering and Preprocessing** | Drops the ID column, converts text categories to numbers, and fills missing values with the column median so the models can use every row. |
| **6. Model Training and Evaluation** | Splits the data 80/20 (stratified so both sets keep the same churn ratio), then trains Logistic Regression, Random Forest, and XGBoost. Results are compared in a table, a bar chart, ROC curves, and an XGBoost feature importance chart. |
| **7. Business Proposal** | Turns the findings into a tiered intervention plan, a device upgrade campaign, an illustrative revenue impact estimate, and a list of limitations and next steps. |

---

## Methodology

**Models compared**

| Model | Role |
| --- | --- |
| Logistic Regression | Simple linear baseline |
| Random Forest (100 trees) | Ensemble of decision trees |
| XGBoost (500 trees, depth 6, learning rate 0.03) | Gradient boosting, best performing model |

**Evaluation:** The data is split 80% for training and 20% for testing with a fixed random seed (42) so results can be reproduced. AUC ROC is the primary metric because it measures how well a model ranks churners above non churners across all decision thresholds. Accuracy and F1 score are reported alongside it.

---

## Results

| Model | Accuracy | AUC ROC | F1 Score |
| --- | --- | --- | --- |
| Logistic Regression | 0.582 | 0.611 | 0.567 |
| Random Forest | 0.620 | 0.669 | 0.615 |
| **XGBoost** | **0.639** | **0.695** | **0.640** |

XGBoost performed best on every metric. A score of 0.5 AUC would be random guessing, so the models find real signal, but there is clear room to improve (see Limitations).

---

## Key Findings

From the exploratory analysis:

* **Older devices:** Customers with older handsets churn at a rate noticeably above the average.
* **Tenure:** Customers in the middle of their tenure (roughly 11 to 16 months) show the highest churn, which suggests a point where people reassess their plan.
* **Usage trends:** Customers whose monthly usage drops sharply churn more, which points to disengagement before leaving.

The feature importance chart in Section 6 shows which fields the XGBoost model relies on most.

---

## Business Proposal

1. **Weekly risk scoring:** Use the XGBoost model to give every customer a churn probability each week.
2. **Tiered action plan:**

| Risk Tier | Score | Recommended Action |
| --- | --- | --- |
| High | Above 70% | Proactive call with a device upgrade offer |
| Medium | 40% to 70% | Targeted SMS or email with a loyalty discount |
| Low | Below 40% | Standard newsletter and service check in |

3. **Device upgrade campaign:** Offer upgrade incentives to customers with older devices who also show declining usage.

The notebook includes an illustrative revenue estimate. It depends on stated assumptions (share of customers targeted, share successfully retained, and revenue per customer), so treat it as a planning example to be refined with Company A's real figures rather than a forecast.

---

## Limitations and Next Steps

* The AUC ROC of about 0.70 leaves room for improvement. Interaction features and rolling usage windows could help.
* The balanced sample means probabilities should be recalibrated before being used in a real deployment.
* The model should be retrained regularly on fresh data to capture seasonal patterns.
* A proper A/B test against a control group is needed to measure the true effect of the retention actions.
* Adding network quality data, such as dropped call rates and coverage complaints, could improve predictions.

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Google Colab

---

## References

1. Bain & Company (2006). *Retaining customers is the real challenge.* https://www.bain.com/insights/retaining-customers-is-the-real-challenge/
2. Verizon Communications (2024, October 22). *Verizon delivers strong third quarter results.* https://www.verizon.com/about/news/verizon-delivers-strong-third-quarter-results-customer-growth-mobility-extending-industry
3. Reuters (2024, July 24). *AT&T beats estimates for subscriber additions.* https://www.reuters.com/business/media-telecom/att-beats-estimates-subscriber-additions-demand-higher-priced-plans-2024-07-24/
4. Statista (2025). *T-Mobile US statistics and facts.* https://www.statista.com/topics/996/t-mobile-us/

---

## Author

Karen John
