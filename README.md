# Agri Companion Advisor – Smart Crop Recommendation System

This project uses machine learning and agricultural intelligence to recommend the best companion crops, irrigation methods, fertilizer needs, and more — all through a smart decision-support system for farmers and agronomists.

It’s designed to enhance sustainable farming by leveraging data science and domain knowledge to ensure optimal crop combinations and efficient resource usage.

---

## About the Project

The Companion Crop Recommendation System is a Python-based application that:

- Suggests **compatible companion crops** for a given main crop  
- Advises **which crops to avoid planting nearby**  
- Recommends **irrigation methods** based on environmental data  
- Estimates **daily water requirements**  
- Provides **land allocation percentages**  
- Suggests **fertilizer needs**  
- Identifies **preferred growing season and soil type**

It blends **agronomic knowledge, environmental sensors**, and **ML algorithms** to empower farmers with smarter crop planning.

---

## Dataset Used

1. `companion_plants_veg.csv`  
2. `agriculture_dataset.csv` 

---

## Tech Stack

| Component           | Technology Used                     |
|---------------------|--------------------------------------|
| **Programming**     | Python 3.x                           |
| **Data Handling**   | Pandas, NumPy                        |
| **ML Modeling**     | Scikit-learn (Random Forest)         |
| **Preprocessing**   | OneHotEncoder, StandardScaler        |
| **NLP Processing**  | NLTK (WordNet Lemmatizer)            |
| **Serialization**   | Joblib                               |
| **Presentation**    | Tabulate (for clean output tables)   |

---

## Alignment with UN Sustainable Development Goals (SDGs)

This project actively supports the following SDGs:

| SDG Goal | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| **#2 – Zero Hunger**        | Promotes efficient, smart farming to increase crop yields and reduce waste |
| **#12 – Responsible Consumption & Production** | Encourages sustainable agricultural practices, optimized input use     |
| **#13 – Climate Action**    | Recommends efficient irrigation to reduce water overuse and preserve resources |
| **#15 – Life on Land**      | Protects soil biodiversity by advising companion planting and avoiding harmful combinations |
| **#9 – Industry, Innovation, and Infrastructure** | Uses technology and innovation to modernize agricultural practices     |

---


## Unique Features

- **AI-Driven**: Predicts best companion crops using ML  
- **Contextual Inputs**: Considers soil moisture, temperature, pH  
- **Fallback Logic**: Smart defaults when crop is unknown  
- **Visual Output Tables**: Tabular summaries for decisions  
- **Resource-Aware**: Suggests irrigation and fertilizer based on actual needs  
- **Intelligent Allocator**: Provides land split % based on crop compatibility  
- **User-Friendly**: CLI interface with clear prompts and validations  

---

## Future Improvements

- Build an interactive web app using Flask or Streamlit  
- Create a mobile application for farmers in remote areas  
- Integrate IoT sensor data (real-time temp/moisture)  
- Add multi-language support (regional languages)  
- Enable CSV or Excel export of recommendations  
- Include crop rotation and pest control suggestions  
- Add visualizations for irrigation/fertilizer/land use graphs  

---
