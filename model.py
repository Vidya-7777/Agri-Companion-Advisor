import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from typing import Dict, List
from difflib import get_close_matches

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# File paths
COMPANION_CSV = "/companion_plants_veg.csv"
AGRI_CSV = "/agriculture_dataset.csv"
MODEL_FILE = "/companion_model.joblib"

# Constants
DRIP_CROPS = {"tomato", "brinjal", "eggplant", "pepper", "capsicum", "chili"}
FLOODED_CROPS = {"rice"}
BROAD_IRRIGATION_CROPS = {"corn", "wheat", "millet", "sorghum"}
MOISTURE_LOVERS = {"lettuce", "spinach", "celery"}
WATER_NEEDS = {
    "tomato": 1.5, "lettuce": 1.0, "pepper": 1.2, "carrot": 1.0,
    "cabbage": 1.3, "brinjal": 1.5, "spinach": 0.8, "corn": 2.0,
    "wheat": 1.6, "onion": 1.1, "strawberry": 1.4, "asparagus": 1.2
}

# Fallback soil type mapping
SOIL_FALLBACK = {
    "carrot": "Loose, well-drained sandy loam",
    "onion": "Loose, well-drained sandy loam",
    "lettuce": "Rich, moist loam",
    "tomato": "Loamy or sandy loam",
    "bean": "Loamy or sandy loam",
    "passion fruit": "Well-drained sandy loam",
    "broccoli": "Fertile, well-drained loam",
    "brassica": "Fertile, well-drained loam",
    "strawberry": "Well-drained sandy loam",
    "asparagus": "Sandy, well-drained soil"
}

# Fallback fertilizer (tons/ha) by crop type
FERT_FALLBACK = {
    "root": 0.025,
    "leafy": 0.015,
    "fruiting": 0.018,
    "berry": 0.022,
    "general": 0.02
}

def normalize_crop_name(name: str) -> str:
    return lemmatizer.lemmatize(name.strip().lower())

def validate_input(prompt: str, min_val: float, max_val: float) -> float:
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"❌ Error: Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print("❌ Error: Please enter a valid number.")

def load_companion_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        crop = normalize_crop_name(str(row["Common name"]))
        for col, is_good in [("Helps", 1), ("Avoid", 0)]:
            if pd.notna(row.get(col)):
                for comp in str(row[col]).split(","):
                    companion = normalize_crop_name(comp)
                    if companion:
                        records.append({"crop_a": crop, "crop_b": companion, "compatible": is_good})
    return pd.DataFrame(records)

def load_agriculture_data(csv_path: str) -> pd.DataFrame:
    """Load agriculture dataset with error handling"""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Warning: Could not load agriculture dataset: {str(e)}")
        return pd.DataFrame()

def build_feature_pipeline(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include="object").columns.difference(["compatible"])
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["compatible"])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

def train_model(df: pd.DataFrame):
    y = df["compatible"]
    X = df.drop(columns=["compatible"])
    pipe = Pipeline([
        ("prep", build_feature_pipeline(df)),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_FILE)
    return pipe

def recommend_companions(crop: str, sensor: Dict, weather: Dict, top_k=5):
    df = load_companion_data(COMPANION_CSV)
    model = joblib.load(MODEL_FILE)
    crop_norm = normalize_crop_name(crop)
    subset = df[df["crop_a"] == crop_norm].copy()

    if subset.empty:
        # Return default recommendations for unknown crops
        companions = [
            {"companion": "Tomato", "score": 0.95},
            {"companion": "Lettuce", "score": 0.93},
            {"companion": "Bean", "score": 0.92},
            {"companion": "Broccoli", "score": 0.91},
            {"companion": "Onion", "score": 0.90}
        ]
        avoid = ["Cabbage", "Cauliflower"]
        return companions, avoid

    for k, v in sensor.items():
        subset.loc[:, k] = v
    for k, v in weather.items():
        subset.loc[:, k] = v
    X = subset.drop(columns=["compatible"])
    subset.loc[:, "score"] = model.predict_proba(X)[:, 1]
    top = subset[subset["compatible"] == 1].sort_values("score", ascending=False).head(top_k)
    avoid = subset[subset["compatible"] == 0]["crop_b"].unique().tolist()
    return top[["crop_b", "score"]].rename(columns={"crop_b": "companion"}).to_dict("records"), avoid

def get_crop_irrigation(crop, temp, moist, pH):
    crop_norm = normalize_crop_name(crop)
    if crop_norm in DRIP_CROPS:
        return "Drip irrigation with mulch"
    elif crop_norm in FLOODED_CROPS:
        return "Alternate Wetting and Drying (AWD)"
    elif crop_norm in BROAD_IRRIGATION_CROPS:
        return "Furrow or basin irrigation"
    elif crop_norm in MOISTURE_LOVERS:
        return "Sprinkler or drip with shading"
    elif temp > 35 and moist < 30:
        return "Sprinkler to reduce evapotranspiration"
    elif pH < 5.5 or pH > 7.5:
        return "Correct pH, then use drip or furrow"
    return "Furrow or drip irrigation"

def estimate_water_requirement(crop, temp, moist, soil="loam"):
    crop_norm = normalize_crop_name(crop)
    base = WATER_NEEDS.get(crop_norm, 1.2)
    adjustment = (1 + (temp - 25) * 0.03) * (1 + max(0, (50 - moist) * 0.01))
    return round(base * adjustment, 2)

def suggest_group_irrigation(crops, temp, moist, pH):
    methods = [get_crop_irrigation(c, temp, moist, pH).lower() for c in crops]
    if all("drip" in m for m in methods):
        return "✅ Drip irrigation suits all crops."
    if all("furrow" in m or "basin" in m for m in methods):
        return "✅ Furrow or basin irrigation fits the group."
    return "⚠️ Mixed irrigation needs — consider grouping differently."

def get_preferable_season(crop, csv_path=AGRI_CSV) -> str:
    try:
        df = pd.read_csv(csv_path)
        crop_col = next((c for c in df.columns if "crop" in c.lower()), None)
        season_cols = [c for c in df.columns if "season" in c.lower()]
        if not crop_col or not season_cols:
            return "Unknown"
        crop_norm = normalize_crop_name(crop)
        df[crop_col] = df[crop_col].str.lower()
        match = df[df[crop_col].str.contains(crop_norm)]
        for col in season_cols:
            val = match[col].dropna().astype(str).iloc[0]
            if val:
                return val.title()
    except:
        pass
    return "Unknown"

def get_recommended_soil(crops: List[str], agri_df: pd.DataFrame) -> str:
    """Get soil recommendation from agriculture dataset with fallback"""
    try:
        soils = []
        for crop in crops:
            crop_norm = normalize_crop_name(crop)
            # Find matching crops in agriculture dataset
            matches = agri_df[
                agri_df['Crop_Type'].str.lower().str.contains(crop_norm, na=False)
            ]
            if not matches.empty:
                # Use the most common soil type for this crop
                soil_counts = matches['Soil_Type'].value_counts()
                if not soil_counts.empty:
                    soils.append(soil_counts.idxmax())

        if soils:
            # Return the most common soil across all crops
            return pd.Series(soils).value_counts().idxmax().title()

        # Fallback to predefined soil types if no matches found
        fallback_soils = []
        for crop in crops:
            crop_norm = normalize_crop_name(crop)
            if crop_norm in SOIL_FALLBACK:
                fallback_soils.append(SOIL_FALLBACK[crop_norm])

        if fallback_soils:
            return pd.Series(fallback_soils).value_counts().idxmax().title()

        return "Well-drained loam"
    except Exception as e:
        print(f"⚠️ Soil recommendation error: {str(e)}")
        return "Well-drained loam"

def allocate_land_share(main_crop: str, companions: List[Dict]) -> pd.DataFrame:
    crops = [main_crop] + [c["companion"] for c in companions]
    scores = {c["companion"]: c["score"] for c in companions}
    scores[main_crop] = 1.0
    df = pd.DataFrame({"Crop": [c.title() for c in crops]})
    df["Score"] = df["Crop"].apply(lambda x: scores.get(x.lower(), 0.5))
    df["Share (%)"] = round(df["Score"] / df["Score"].sum() * 100, 2)
    return df[["Crop", "Share (%)"]]

def infer_fert_type(crop: str) -> str:
    crop_lower = crop.lower()
    if any(x in crop_lower for x in ["carrot", "onion", "radish", "beet", "asparagus"]):
        return "root"
    elif any(x in crop_lower for x in ["lettuce", "spinach", "cabbage", "broccoli", "brassica"]):
        return "leafy"
    elif any(x in crop_lower for x in ["tomato", "bean", "pepper", "eggplant", "brinjal", "chili", "capsicum", "passion fruit"]):
        return "fruiting"
    elif any(x in crop_lower for x in ["strawberry", "raspberry", "blueberry"]):
        return "berry"
    else:
        return "general"

def get_fertilizer_recommendation_by_area(crop: str, area_ha: float) -> float:
    fert_type = infer_fert_type(crop)
    fallback_rate = FERT_FALLBACK.get(fert_type, FERT_FALLBACK["general"])
    return round(fallback_rate * area_ha, 2)

# 🌿 Main interaction
if __name__ == "__main__":
    # Load datasets
    df_train = load_companion_data(COMPANION_CSV)
    agri_df = load_agriculture_data(AGRI_CSV)
    model = train_model(df_train)

    user_crop = input("\n🌿 Enter a crop name: ").strip()

    # Input validation
    moisture = validate_input("💧 Soil moisture (0–100%): ", 0, 100)
    ph = validate_input("🧪 Soil pH (3.5–9.0): ", 3.5, 9.0)
    temp = validate_input("🌡️ Temperature (5–50 °C): ", 5, 50)

    try:
        companions, avoid = recommend_companions(
            user_crop,
            sensor={"soil_moisture": moisture, "soil_pH": ph},
            weather={"temp": temp},
            top_k=5
        )

        all_crops = [user_crop] + [c["companion"] for c in companions]
        table_data = []

        print(f"\n🌱 Recommended companions for '{user_crop.title()}':")
        for i, s in enumerate(companions, 1):
            print(f"{i}. {s['companion'].title():<20}  Score = {s['score']:.2f}")

        if avoid:
            print(f"\n🚫 Avoid planting near '{user_crop.title()}':")
            for crop in avoid:
                print(f"• {crop.title()}")
        else:
            print(f"\n✅ No avoid-crops listed for '{user_crop.title()}'.")

        print(f"\n📊 Irrigation & Water Requirement Summary:")
        for crop in all_crops:
            method = get_crop_irrigation(crop, temp, moisture, ph)
            water = estimate_water_requirement(crop, temp, moisture)
            table_data.append([crop.title(), method, f"{water:.2f} L/day"])

        print(tabulate(table_data, headers=["Crop", "Irrigation Method", "Water Needed"], tablefmt="fancy_grid"))

        group_method = suggest_group_irrigation(all_crops, temp, moisture, ph)
        print(f"\n🌍 Group Recommendation:\n{group_method}")

        # 📐 Land Allocation
        land_df = allocate_land_share(user_crop, companions)
        print(f"\n📐 Suggested Land Allocation (% of Total Land):")
        print(tabulate(land_df, headers="keys", tablefmt="fancy_grid"))

        # 📅 Preferable Season
        season_list = [get_preferable_season(crop) for crop in all_crops if get_preferable_season(crop) != "Unknown"]
        if season_list:
            best_season = pd.Series(season_list).value_counts().idxmax()
            print(f"\n📅 Preferable Season for this crop combination: {best_season}")
        else:
            print("\n📅 Preferable Season for this crop combination: Unknown")

        # 🧱 Recommended Soil Type - UPDATED TO USE DATASET
        best_soil = get_recommended_soil(all_crops, agri_df)
        print(f"\n🧱 Recommended Soil Type for this crop combination: {best_soil}")

        # 🌾 Fertilizer Recommendation
        print(f"\n🌾 Fertilizer Recommendation Based on Area Usage:")
        for _, row in land_df.iterrows():
            crop = row["Crop"]
            share_percent = row["Share (%)"]
            hectares = 1.0 * share_percent / 100  # total 1 hectare assumed
            fert = get_fertilizer_recommendation_by_area(crop, hectares)
            if fert > 0:
                print(f"• {crop}: Apply approx. {fert:.2f} tons over {hectares:.2f} ha")
            else:
                print(f"• {crop}: Apply a balanced NPK fertilizer as per local recommendations")

    except ValueError as ve:
        print("\n❌ Error:", ve)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
