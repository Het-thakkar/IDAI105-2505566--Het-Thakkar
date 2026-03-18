# 🛍️ Black Friday Sales Insights
## Data Mining Summative Assessment — Scenario 1
 
> **CRS:** Artificial Intelligence | **Course:** Data Mining | **Marks:** 60
 
---
 
## 📌 Project Overview
 
This project analyzes the **Black Friday retail sales dataset** to uncover actionable shopping insights using data mining techniques. As a Data Analyst at *Insight Mart Analytics*, the goal is to understand customer purchase behavior, segment shoppers, discover product associations, and detect anomalies.
 
**Live Streamlit App:** [🔗 Click here to view the dashboard](#) *(replace with your deployed link)*
 
---
 
## 🗂️ Repository Structure
 
```
IDAI105(StudentID)-YourName/
│
├── BlackFriday_Analysis.ipynb   # Full analysis notebook (Stages 1–8)
├── app.py                       # Streamlit dashboard app
├── requirements.txt             # Python dependencies
├── BlackFriday.csv              # Dataset (download from assignment link)
│
├── plots/                       # Generated visualizations
│   ├── plot_purchase_distribution.png
│   ├── plot_age_purchase.png
│   ├── plot_category_sales.png
│   ├── plot_heatmap.png
│   ├── plot_elbow.png
│   ├── plot_clusters.png
│   ├── plot_association_rules.png
│   ├── plot_anomalies.png
│   └── plot_anomaly_demographics.png
│
└── README.md
```
 
---
 
## 📊 Dataset
 
**Source:** Black Friday Sales Dataset  
**Download:** [Google Drive Link](https://drive.google.com/drive/folders/13DxtCVj3S_AAYXG5THw2mmr6_VA1N3L9)
 
| Column | Description |
|---|---|
| User_ID | Unique customer identifier |
| Product_ID | Unique product identifier |
| Gender | M / F |
| Age | Age group (0-17, 18-25, 26-35, ...) |
| Occupation | Occupation code (0–20) |
| City_Category | City tier (A, B, C) |
| Stay_In_Current_City_Years | Years in current city |
| Marital_Status | 0 = Single, 1 = Married |
| Product_Category_1/2/3 | Product category codes |
| Purchase | Purchase amount in USD |
 
---
 
## 🔧 Key Preprocessing Steps
 
- **Missing values:** `Product_Category_2` and `Product_Category_3` filled with `0`
- **Encoding:** Gender → (M=0, F=1); Age groups → ordered integers (1–7); City → Label Encoded
- **Normalization:** Purchase amounts scaled using `StandardScaler`
- **Duplicates:** Removed all duplicate rows
- **Data type fixes:** `Stay_In_Current_City_Years` — `'4+'` replaced with `4`
 
---
 
## 📈 EDA Visualizations
 
- **Histogram** — Distribution of purchase amounts
- **Box Plots** — Purchase by Gender and Age group
- **Bar Charts** — Top product categories by total revenue
- **Scatter Plot** — Purchase vs Occupation
- **Correlation Heatmap** — Relationships between all key features
 
---
 
## 🔵 Clustering Analysis
 
**Algorithm:** K-Means  
**Optimal K:** 4 (determined via Elbow Method)  
**Features Used:** Age, Gender, Occupation, Marital Status, Normalized Purchase
 
| Cluster | Label | Description |
|---|---|---|
| 0 | 💸 Discount Lovers | Low-spend, deal-seeking customers |
| 1 | 👑 Premium Buyers | High-spend, quality-focused shoppers |
| 2 | 🛒 Casual Shoppers | Moderate, infrequent buyers |
| 3 | 🔁 Frequent Buyers | Regular shoppers with consistent spending |
 
Clusters visualized in 2D using **PCA** projection.
 
---
 
## 🔗 Association Rule Mining
 
**Algorithm:** Apriori  
**Parameters:** min_support=0.05, min_confidence=0.3, min_lift=1.2
 
Finds product categories frequently purchased together.  
Rules evaluated using **Support**, **Confidence**, and **Lift** metrics.
 
Example rule: *"If a customer buys Category 1, they are likely to also buy Category 5"*
 
---
 
## 🚨 Anomaly Detection
 
**Methods Used:** Z-Score (threshold=3) and IQR (multiplier=1.5)
 
- Computes **total spending per user**
- Flags statistically unusual spenders
- Compares anomalies by Gender, Age, and Occupation
- Useful for identifying VIP customers for special loyalty programs
 
---
 
## 🚀 Streamlit App Features
 
The deployed app includes 5 interactive pages:
 
| Page | Content |
|---|---|
| 📊 Overview & EDA | KPIs, purchase by gender/age/city, heatmap, box plots |
| 🔵 Customer Clustering | Elbow curve, PCA scatter, cluster stats table |
| 🔗 Association Rules | Interactive sliders, rules table, support vs confidence scatter |
| 🚨 Anomaly Detection | Method toggle, highlighted histogram, demographic breakdown |
| 💡 Insights Summary | Key findings, occupation trends, marital status analysis |
 
---
 
## ⚙️ How to Run Locally
 
```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/IDAI105(StudentID)-YourName.git
cd IDAI105(StudentID)-YourName
 
# 2. Install dependencies
pip install -r requirements.txt
 
# 3. Run the Streamlit app
streamlit run app.py
 
# 4. Or open the notebook
jupyter notebook BlackFriday_Analysis.ipynb
```
 
---
 
## 🌐 Deploying on Streamlit Cloud
 
1. Push all files to your GitHub repository
2. Visit [https://share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New App** → select your repo → set `app.py` as main file
5. Click **Deploy** and share the generated link
 
---
 
## 📚 References
 
- [Data-to-Viz: Chart Types](https://www.data-to-viz.com/)
- [K-Means Clustering Guide](https://neptune.ai/blog/k-means-clustering)
- [Market Basket Analysis](https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-market-basket-analysis/)
- [Anomaly Detection in Python](https://www.datacamp.com/courses/anomaly-detection-in-python)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Streamlit Documentation](https://docs.streamlit.io)
 
---
 
## 👤 Student Details
 
| Field | Details |
|---|---|
| **Full Name** | *(Het Thakkar)* |
| **CRS** | Artificial Intelligence |
| **School** | *(Udgam school for children)* |
 
