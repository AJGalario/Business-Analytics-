# Phase 1: Business Problem & Dataset Selection
**Course Project:** Business Analytics Final Project  
**Folder:** `Final/`  

---

## 1. Business Domain & Context

### The Domain: E-Commerce Marketplace (Olist)
This project focuses on the **Brazilian E-Commerce Marketplace** domain, analyzing real-world transactional data provided by **Olist**, the largest department store marketplace in Brazilian e-commerce. Olist operates as an integrator platform: it connects small, independent local businesses across Brazil with major e-commerce retail channels (such as Submarino, Americanas, and Mercado Livre) through a single dashboard. 

---

## 2. Chosen Dataset Specifications

Our analysis utilizes the official **Brazilian E-Commerce Public Dataset by Olist** (sourced from Kaggle). The original 9 relational tables were cleaned, aggregated, and joined into a single consolidated, granular item-level dataset:

* **Dataset Filename:** `olist_granular_dataset.csv`
* **Format:** Comma-Separated Values (`.csv`) / Excel (`.xlsx`)
* **Number of Records (Rows):** **112,650 rows** (representing every individual item sold).
* **Number of Attributes (Columns):** **32 variables** (capturing order details, customer demographics, seller geography, product dimensions, pricing, and customer feedback).
* **Data Granularity:** **Order-Item Level**. A unique row in this dataset represents one individual item within a customer's order. If an order contains three items, it will occupy three distinct rows sharing the same `Order ID`.

---

## 3. Data Dictionary

The following table serves as our official Data Dictionary, defining each of the 32 columns in our consolidated analytical dataset:

| # | Column Name | Type | Business Description | Example Values |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `order_id` | String | Unique order identifier | 00010242fe... |
| **2** | `order_item_id` | Integer | Sequence number of item within order | 1, 2, 3 |
| **3** | `order_status` | String | Current status of the order | delivered, canceled |
| **4** | `order_purchase_timestamp` | DateTime | When the order was placed | 2017-09-13 08:59:02 |
| **5** | `order_approved_at` | DateTime | When payment was approved | 2017-09-13 09:45:35 |
| **6** | `order_delivered_carrier_date` | DateTime | When shipped to carrier | 2017-09-19 18:34:16 |
| **7** | `order_delivered_customer_date` | DateTime | When delivered to customer | 2017-09-20 23:43:48 |
| **8** | `order_estimated_delivery_date` | DateTime | Estimated delivery date | 2017-09-29 00:00:00 |
| **9** | `customer_id` | String | Unique customer identifier | 3ce436f1... |
| **10** | `customer_unique_id` | String | Unique ID per individual | 871766c5... |
| **11** | `customer_zip_code_prefix` | Integer | Customer ZIP prefix | 28013 |
| **12** | `customer_city` | String | Customer city name | campos dos goytacazes |
| **13** | `customer_state` | String | Customer state (2-letter) | RJ, SP, MG |
| **14** | `product_id` | String | Unique product identifier | 4244733e... |
| **15** | `product_category_name` | String | Product category (Portuguese) | cool_stuff |
| **16** | `product_category_name_english` | String | Product category (English) | cool_stuff |
| **17** | `seller_id` | String | Unique seller identifier | 48436dad... |
| **18** | `seller_city` | String | Seller city | volta redonda |
| **19** | `seller_state` | String | Seller state (2-letter) | SP |
| **20** | `price` | Float | Price of this item (BRL) | 58.90 |
| **21** | `freight_value` | Float | Shipping cost for this item (BRL) | 13.29 |
| **22** | `payment_type` | String | Payment method | credit_card, boleto |
| **23** | `payment_value` | Float | Total payment amount for item (BRL) | 72.19 |
| **24** | `avg_review_score` | Float | Average review score (1–5) | 5.0, 4.0 |
| **25** | `num_reviews` | Integer | Number of reviews received | 1 |
| **26** | `product_name_lenght` | Integer | Characters in product name | 58 |
| **27** | `product_description_lenght` | Integer | Characters in product description | 598 |
| **28** | `product_photos_qty` | Integer | Number of product photos | 4 |
| **29** | `product_weight_g` | Float | Product weight in grams | 650.0 |
| **30** | `product_length_cm` | Float | Product length in cm | 28.0 |
| **31** | `product_height_cm` | Float | Product height in cm | 9.0 |
| **32** | `product_width_cm` | Float | Product width in cm | 14.0 |

---

## 4. Business Questions (DeepSeek Adopted Questions)

Here are the 7 e-commerce focused business questions:

### Q1 — Descriptive Statistics
* **Question:** What do typical prices, freight costs, product weights, and review scores look like across Olist's product categories? Which categories are cheap (commoditized) vs. expensive (premium)?
* **Why this matters:** A seller deciding what to sell needs to know if a category has room for a high-priced product or if it's a race to the bottom. Olist can decide where to recruit more sellers.
* **Decision:** "Which categories should I list my products in?" / "Does category X support a premium price?"
* **Visualization:** Histogram or KDE plot

### Q2 — Trend Analysis
* **Question:** Which months of the year have the highest and lowest sales? Do spikes happen around predictable events like Black Friday or Christmas, and which product categories drive them?
* **Why this matters:** Sellers need to know when to stock inventory and run ads. If electronics peak in November and furniture in December, a seller should plan their marketing calendar and cash flow accordingly.
* **Decision:** "When should I launch promotions?" / "How much inventory should I order for each season?"
* **Visualization:** Line chart (time series)

### Q3 — Cross-Tabulation Analysis
* **Question:** Do customers pay differently depending on what they're buying? For example, do people use credit card installments for electronics but boleto (bank slip) for home goods?
* **Why this matters:** If 80% of electronics buyers use 6-installment credit cards, removing that option kills sales. If budget categories lean toward boleto, offering a boleto discount could capture more customers.
* **Decision:** "Which product categories need installment payment options?" / "Should I offer boleto discounts on specific items?"
* **Visualization:** Heatmap (crosstab)

### Q4 — Drill-Down Analysis
* **Question:** Which states and cities have the worst delivery delays? And are those problem areas big revenue markets or small ones?
* **Why this matters:** Olist has to decide where to invest in logistics. Fixing a city that represents 1% of revenue but has 50% of delays is a waste. Fixing a city with high revenue AND long delays is urgent.
* **Decision:** "Which 3 cities should I fix carrier contracts for first?" / "Where should Olist build its next fulfillment center?"
* **Visualization:** Hierarchical bar or treemap

### Q5 — Correlation & Predictive Analysis
* **Question:** What actually makes customers give better reviews — lower prices, faster shipping, or better product photos? If a seller has 100 BRL to spend on improving their ratings, where should they put it?
* **Why this matters:** Sellers have limited money. They can cut prices, subsidize shipping, or improve their product pages. A regression model tells them which move has the biggest impact on satisfaction for the same cost.
* **Decision:** "Should I cut my price by 10 BRL or offer free shipping?" / "Is it worth hiring a photographer for better product photos?"
* **Visualization:** Scatter plot with regression line

### Q6 — Ad Hoc (Follow-up from Q5)
* **Question:** Are there product categories where customers happily pay high prices AND give 5-star reviews? What do those categories have in common?
* **Why this matters:** If some categories support premium pricing without hurting satisfaction, Olist can target those for seller recruitment and premium positioning.
* **Decision:** "Which categories should I launch a premium product line in?"
* **Visualization:** Interactive Bubble Scatter Plot (Average Price vs. % 5-Star Reviews)

### Q7 — Ad Hoc (Follow-up from Q4)
* **Question:** Are the regions with the longest delivery delays also paying the highest shipping costs? In other words, are some customers getting a raw deal — paying more and waiting longer?
* **Why this matters:** If remote customers pay premium shipping and still get bad service, they'll stop buying. Olist needs to know whether to subsidize those routes or renegotiate carrier contracts.
* **Decision:** "Should we subsidize shipping to certain states?" / "Which carrier routes need renegotiation?"
* **Visualization:** Interactive Bubble Scatter Plot (Average Shipping Fee vs. Average Delay by State)
