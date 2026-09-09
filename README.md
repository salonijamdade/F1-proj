# 🤖 Review Intelligence — Agentic AI

> **Transform customer reviews into actionable business insights using Unsupervised Machine Learning, NLP, Anomaly Detection, and Multi-Agent AI.**

Review Intelligence is an interactive Streamlit application that automatically analyses customer reviews without requiring pre-labelled data. Upload a CSV file containing customer reviews and discover hidden patterns, customer groups, recurring topics, suspicious reviews, product risks, and business recommendations.

---

## ✨ Features

### 📊 Unsupervised Review Analysis

* Automatically groups similar customer reviews
* No manually labelled dataset required
* Automatically finds the optimal number of clusters
* Analyses customer satisfaction patterns

### 🔍 Multiple Clustering Algorithms

The application compares three clustering approaches:

* **K-Means Clustering** — Primary clustering algorithm
* **DBSCAN** — Density-based clustering with noise detection
* **Agglomerative Clustering** — Hierarchical clustering for relationship analysis

### 🧩 Topic Modeling

Automatically discovers recurring themes from customer reviews using:

* **NMF (Non-Negative Matrix Factorization)**
* **LDA (Latent Dirichlet Allocation)**

Example themes:

* 🔋 Battery Issues
* 📷 Camera Quality
* ⚡ Performance
* 🐌 Performance Issues
* 💰 Value for Money
* 🔥 Heating Problems
* 📱 Display Quality
* 📦 Delivery & Packaging
* 💻 Software/UI

---

## ⚠️ Anomaly Detection

The system identifies unusual or suspicious reviews using:

* **Isolation Forest**
* **Local Outlier Factor (LOF)**

Potential anomalies may include:

* Fake or spam reviews
* Extremely unusual reviews
* One-off product defects
* Reviews that significantly differ from common customer patterns

---

## 🧠 Multi-Agent AI System

The application uses multiple specialised AI agents:

| Agent                  | Role                                               |
| ---------------------- | -------------------------------------------------- |
| 📊 Data Analyst        | Analyses customer clusters and identifies patterns |
| ⚠️ Risk Analyst        | Detects product issues and customer pain points    |
| 💡 Business Strategist | Provides actionable business recommendations       |
| 📝 Report Writer       | Generates executive-level reports                  |
| 🧠 Orchestrator        | Synthesises multiple agent responses               |

The system automatically selects the appropriate specialist based on the user's question.

### Example Questions

* What are the biggest problems?
* Which customer cluster needs attention?
* How do verified buyers differ?
* Give me an executive report
* What product issues are customers complaining about?
* What should we improve?

---

## 📈 Visualisations

The dashboard includes:

* 📍 Cluster Scatter Plot
* 📊 Cluster Size Comparison
* 📉 Elbow Method
* 📏 Silhouette Score Analysis
* 🧩 Topic Prevalence Chart
* ⭐ Rating Distribution
* 🌳 Hierarchical Clustering Dendrogram
* ⚠️ Anomaly Detection Graph

---

## 📁 Dataset Format

Upload a CSV file containing at least the following column:

```csv
review_text
```

### Optional Columns

```csv
review_text,rating,verified_purchase,product_variant,date
```

Example:

```csv
review_text,rating,verified_purchase,product_variant,date
"The battery life is amazing and the camera is great",5,Yes,128GB,2026-01-10
"The phone heats up while gaming",2,Yes,256GB,2026-01-11
"Good value for money",4,No,128GB,2026-01-12
```

### Required Column

| Column        | Description          |
| ------------- | -------------------- |
| `review_text` | Customer review text |

### Optional Columns

| Column              | Description                       |
| ------------------- | --------------------------------- |
| `rating`            | Customer rating                   |
| `verified_purchase` | Whether the purchase was verified |
| `product_variant`   | Product variant                   |
| `date`              | Date of review                    |

---

## ⚙️ Machine Learning Pipeline

The application follows this workflow:

```text
                ┌─────────────────┐
                │   Upload CSV    │
                │ Customer Reviews│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Data Cleaning   │
                │ Remove Duplicates│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Feature Building│
                │     TF-IDF      │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Unsupervised Learning│
              ├──────────────────────┤
              │ K-Means              │
              │ DBSCAN               │
              │ Agglomerative        │
              └──────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Topic Modeling  │
                │ NMF / LDA       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Anomaly Detection│
                │ Isolation Forest │
                │ LOF              │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Multi-Agent AI  │
                │ Business Insights│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Executive Report│
                │ & Recommendations│
                └─────────────────┘
```

---

## 🛠️ Technologies Used

### Frontend

* [Streamlit](https://streamlit.io)

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Scipy

### NLP

* TF-IDF Vectorization
* Count Vectorization
* NMF
* LDA
* Truncated SVD

### AI

* Multi-Agent Architecture
* LLM API Integration
* AI Orchestration

### Visualisation

* Matplotlib
* PCA
* t-SNE
* Dendrograms

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/review-intelligence.git
```

### 2. Navigate to the Project

```bash
cd review-intelligence
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib scipy scikit-learn requests
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will start locally in your browser.

---

## ⚙️ Configurable Settings

The dashboard allows users to customise:

### Analysis Settings

* TF-IDF vocabulary size
* Cluster count range
* Number of topics
* Topic modeling method
* PCA or t-SNE visualisation
* Anomaly detection sensitivity

### Alert Thresholds

Configure alerts based on:

* Maximum anomaly percentage
* Minimum average rating
* Minimum cluster rating
* Verified vs unverified rating gap

---

## 📊 Output

After analysing the reviews, the application provides:

### Customer Insights

* Total number of reviews
* Average rating
* Number of customer groups
* Cluster quality score
* Topics discovered
* Number of unusual reviews

### Customer Groups

Each cluster includes:

* Cluster size
* Percentage of total reviews
* Average rating
* Verified purchase percentage
* Important keywords
* Sample customer reviews

### AI Insights

AI agents provide:

* Product issue analysis
* Risk assessment
* Business strategies
* Improvement recommendations
* Executive reports

---

## 📥 Export Options

Users can export:

* 📄 **Enriched CSV**
* 📊 **Analysis JSON**
* 📝 **Executive Report in Markdown**

---

## 🎯 Project Objectives

This project aims to demonstrate how **Machine Learning and Agentic AI can work together** to analyse unstructured customer feedback.

The system helps businesses:

* Understand customer sentiment
* Discover hidden customer groups
* Identify product issues
* Detect suspicious reviews
* Find recurring themes
* Make data-driven decisions
* Generate executive reports automatically

---

## 🔮 Future Improvements

* [ ] Sentiment Analysis
* [ ] Real-time review monitoring
* [ ] Review language detection
* [ ] Multilingual review analysis
* [ ] Automatic email alerts
* [ ] Database integration
* [ ] User authentication
* [ ] Cloud deployment
* [ ] Historical trend analysis
* [ ] Product comparison
* [ ] Improved AI agent orchestration

---

## 🧠 Key Concepts Demonstrated

* Unsupervised Machine Learning
* Natural Language Processing
* Feature Engineering
* Clustering
* Topic Modeling
* Anomaly Detection
* Dimensionality Reduction
* Multi-Agent AI
* AI Orchestration
* Data Visualisation
* Streamlit Application Development

---

## 👩‍💻 Author

**Your Name**

Built as a project exploring the combination of:

> **Machine Learning + NLP + Multi-Agent AI + Business Intelligence**

---

## ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!

---

<p align="center">
  Made with ❤️ using Python, Machine Learning and Agentic AI
</p>
