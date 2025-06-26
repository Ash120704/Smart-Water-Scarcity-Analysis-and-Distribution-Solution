
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import requests
from branca.element import Template, MacroElement

st.set_page_config(layout="wide")
st.title("💧 Groundwater Monitoring & Water Demand Prediction")

# --- Select State ---
state = st.selectbox("Select State", ["Telangana", "Andhra Pradesh"])

# Helper function
def classify_scarcity(level):
    if level < 2: return "Very Safe", "#00FF00"
    elif level < 5: return "Safe", "#7FFF00"
    elif level < 10: return "Moderate", "#FFFF00"
    elif level < 20: return "Warning", "#FFA500"
    elif level < 40: return "Critical", "#FF4500"
    return "Severe", "#8B0000"

# --- Telangana Block ---
if state == "Telangana":
    geojson_path = r"C:/Users/Admin/Downloads/TELANGANA_DISTRICTS.geojson"
    groundwater_path = r"C:/Users/Admin/Downloads/may_corrected (1).csv"
    population_path = r"C:/Users/Admin/Downloads/population_corrected (1).csv"
    demand_path = r"C:/Users/Admin/Downloads/demand_estimate_corrected (1).csv"
    line_chart_path = r"C:/Users/Admin/OneDrive/line chart.csv"

    df = pd.read_csv(groundwater_path)

# Optional corrections
    correction_dict = {
        'MEDCHAL': 'MEDCHAL MALKAJGIRI'
    }
    df['district'] = df['district'].replace(correction_dict)
    df['district'] = df['district'].str.upper().str.strip()
    df.rename(columns={'Static Water Level (mbgl)': 'groundwater_level'}, inplace=True)

# Read GeoJSON
    gdf = gpd.read_file(geojson_path)
    gdf['dtname'] = gdf['dtname'].str.upper().str.strip()

# Population
    pop = pd.read_csv(population_path)
    pop.rename(columns={'District': 'district', 'Population': 'population'}, inplace=True)
    pop['district'] = pop['district'].str.upper().str.strip()

# Water demand
    demand = pd.read_csv(demand_path)
    demand.rename(columns={'District': 'district', 'Estimated_Water_Demand_MLD': 'water_demand'}, inplace=True)
    demand['district'] = demand['district'].str.upper().str.strip()

# Clean groundwater data
    df = df.dropna(subset=['groundwater_level'])

# Average groundwater level per district
    df_latest = df.groupby('district', as_index=False)['groundwater_level'].mean()

# Merge with population and demand
    df_latest = df_latest.merge(pop, on='district', how='left')
    df_latest = df_latest.merge(demand, on='district', how='left')
    df_latest = df_latest.dropna()

# ✅ Correct dropdown using df_latest
    selected_district = st.selectbox("Select District", df_latest['district'].sort_values())


    # --- Weather ---
    def get_weather_data(district_name):
        api_key = "782a2f11195ba039b53da13244a2a4fd"
        params = {'q': f"{district_name},IN", 'appid': api_key, 'units': 'metric'}
        try:
            r = requests.get("http://api.openweathermap.org/data/2.5/weather", params=params, timeout=5)
            d = r.json()
            if r.status_code == 200:
                return d['main']['temp'], d['main']['humidity'], d.get('rain', {}).get('1h', 0)
        except: return None, None, None
        return None, None, None

    temp, humidity, rainfall = get_weather_data(selected_district.title())
    st.subheader("🌦️ Current Weather")
    if temp:
        st.metric("Temperature (°C)", f"{temp:.1f}")
        st.metric("Humidity (%)", f"{humidity}%")
        st.metric("Rainfall (mm)", f"{rainfall} mm")
    else:
        st.warning("Live weather unavailable.")

    # --- Groundwater Choropleth ---
    st.subheader("🗺️ Groundwater Scarcity Map")
    df_best = df.sort_values('groundwater_level').drop_duplicates('district')
    pred_levels = []
    for dist in df_best['district'].unique():
        dist_df = df[df['district'] == dist]
        X = pd.DataFrame({'index': range(len(dist_df))})
        y = dist_df['groundwater_level']
        if len(y) >= 2:
            model = GradientBoostingRegressor().fit(X, y)
            pred = model.predict([[len(dist_df)]])[0]
            pred_levels.append({'district': dist, 'predicted_level': pred})
    pred_df = pd.DataFrame(pred_levels)
    merged = gdf.merge(pred_df, left_on="dtname", right_on="district", how="inner")
    m = folium.Map(location=[17.5, 79.0], zoom_start=7)
    for _, row in merged.iterrows():
        level = row['predicted_level']
        status, color = classify_scarcity(level)
        folium.GeoJson(row['geometry'], style_function=lambda _, c=color: {"fillColor": c, "color": "black", "weight": 1, "fillOpacity": 0.7},
                       tooltip=folium.Tooltip(f"{row['dtname'].title()}: {level:.2f} m BGL ({status})")).add_to(m)
    # Groundwater scarcity color classification logic
    color_legend = [
        ("Very Safe", "< 2 m", "#00FF00"),
        ("Safe", "2–5 m", "#7FFF00"),
        ("Moderate", "5–8 m", "#FFFF00"),
        ("Warning", "8–11 m", "#FFA500"),
        ("Critical", "11–15 m", "#FF4500"),
        ("Severe", "> 15 m", "#8B0000")]

    st.subheader("🖍️ Groundwater Level Color Legend")
    for label, range_text, color in color_legend:
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>⬤</span> {label}: {range_text}", unsafe_allow_html=True)

    st_folium(m, width=700)
    # --- Predicted Water Demand ---
    st.subheader("🚰 Predicted Water Demand")
    X_demand = df_latest[['population', 'groundwater_level']]
    y_demand = df_latest['water_demand']
    model = GradientBoostingRegressor().fit(X_demand, y_demand)
    df_latest['predicted_water_demand'] = model.predict(X_demand)
    if selected_district in df_latest['district'].values:
        demand_value = df_latest[df_latest['district'] == selected_district]['predicted_water_demand'].values[0]
        st.success(f"Predicted Water Demand for **{selected_district.title()}**: **{demand_value:.2f} MLD**")
    else:
        st.warning("Water demand prediction not available.")
    st.subheader("🌍 Water Demand Map")
    merged_demand = gdf.merge(df_latest, left_on="dtname", right_on="district", how="inner")
    m2 = folium.Map(location=[17.5, 79.0] if state == "Telangana" else [16.5, 80.5], zoom_start=7, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gdf,
        data=merged_demand,
        columns=["district", "predicted_water_demand"],
        key_on="feature.properties.dtname",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Predicted Water Demand (MLD)"
        ).add_to(m2)
    for _, row in merged_demand.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "black",
                "weight": 0.5
            },
            tooltip=folium.Tooltip(
                f"{row['district'].title()}<br>Demand: {row['predicted_water_demand']:.2f} MLD"
            )
        ).add_to(m2)
    st_folium(m2, width=700)
    import pandas as pd
    import plotly.express as px
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder

# Load your real past groundwater + climate data
    past_data = pd.read_csv(r"C:\Users\Admin\OneDrive\line chart.csv")

# Load current water demand per district
    demand_data = pd.read_csv(r"C:\Users\Admin\Downloads\demand_estimate_corrected (1).csv")

# Load current population per district
    population_data = pd.read_csv(r"C:\Users\Admin\Downloads\population_corrected (1).csv")

# Prepare month list
    future_months = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']

# Create mapping for months to numbers
    month_mapping = {month: idx for idx, month in enumerate(future_months, start=1)}
    past_data['Month_Num'] = past_data['Month_Num'].map(month_mapping)

    past_data['Soil_Recharge_Factor'] = past_data['Soil_Recharge_Factor']

# Merge demand estimates into past_data
    past_data = past_data.merge(demand_data[['District', 'water_demand']], on='District', how='left')
    past_data = past_data.dropna(subset=['water_demand'])

# Merge population estimates into past_data
    past_data = past_data.merge(population_data[['District', 'Population']], on='District', how='left')

# Label encode District
    le = LabelEncoder()
    past_data['District_Code'] = le.fit_transform(past_data['District'])

# --- Groundwater Level Model ---

    X_gwl = past_data[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C']]
    y_gwl = past_data['Groundwater_Level_mBGL']
    model_gwl = GradientBoostingRegressor().fit(X_gwl, y_gwl)

# --- Water Demand Model ---

    X_demand = past_data[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C', 'Soil_Recharge_Factor', 'Groundwater_Level_mBGL']]
    y_demand = past_data['water_demand']
    model_demand = GradientBoostingRegressor().fit(X_demand, y_demand)
    average_rainfall = past_data['Rainfall_mm'].mean()
    average_temp = past_data['Temperature_C'].mean()

# Future Month to Year mapping (for population growth adjustment)
    months_to_years = {
        'January': 0,
        'February': 1/12,
        'March': 2/12,
        'April': 3/12,
        'May': 4/12,
        'June': 5/12,
        'July': 6/12,
        'August': 7/12,
        'September': 8/12,
        'October': 9/12,
        'November': 10/12,
        'December': 11/12}
    population_growth_rate = 0.015  # 1.5% annual growth

# Generate future predictions for each district separately
    future_predictions = []
    for district in past_data['District'].unique():
        district_code = le.transform([district])[0]
        soil_factor = past_data[past_data['District'] == district]['Soil_Recharge_Factor'].iloc[0]
        current_population = past_data[past_data['District'] == district]['Population'].iloc[0]
        future_df = pd.DataFrame({
            'District': [district]*12,
            'District_Code': [district_code]*12,
            'Month': future_months,
            'Month_Num': list(range(1, 13)),
            'Rainfall_mm': [average_rainfall]*12,
            'Temperature_C': [average_temp]*12,
            'Soil_Recharge_Factor': [soil_factor]*12,
            'population': [current_population]*12})

        X_future_gwl = future_df[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C']]
        future_df['Predicted_Groundwater_Level_mBGL'] = model_gwl.predict(X_future_gwl)

    # Predict water demand
        X_future_demand = future_df[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C', 'Soil_Recharge_Factor', 'Predicted_Groundwater_Level_mBGL']]
        X_future_demand = X_future_demand.rename(columns={'Predicted_Groundwater_Level_mBGL': 'Groundwater_Level_mBGL'})
        future_df['Predicted_Water_Demand_MLD'] = model_demand.predict(X_future_demand)
        future_df['Years_into_Future'] = future_df['Month'].map(months_to_years)
        future_df['Future_Population'] = future_df['population'] * ((1 + population_growth_rate) ** future_df['Years_into_Future'])
        future_df['Adjusted_Predicted_Water_Demand_MLD'] = future_df['Predicted_Water_Demand_MLD'] * (future_df['Future_Population'] / future_df['population'])
        future_predictions.append(future_df)
    final_future_predictions = pd.concat(future_predictions, ignore_index=True)

# --- 📈 Plotting Section ---

    st.subheader("📍 Select District to View Future Predictions")
    selected_district = st.selectbox("Select District", final_future_predictions['District'].unique())

    selected_future_df = final_future_predictions[final_future_predictions['District'] == selected_district]

# --- 📈 Plot 1: Groundwater Level Line Chart
    st.subheader(f"📈 Predicted Future Groundwater Level - {selected_district}")

    fig_gwl = px.line(
        selected_future_df,
        x='Month',
        y='Predicted_Groundwater_Level_mBGL',
        markers=True,
        labels={'Predicted_Groundwater_Level_mBGL': 'Groundwater Level (m BGL)'},
        title=f'Groundwater Level Prediction for {selected_district}')
    st.plotly_chart(fig_gwl, use_container_width=True)

# --- 📈 Plot 2: Adjusted Water Demand Line Chart
    st.subheader(f"📈 Predicted Future Water Demand (Adjusted for Population Growth) - {selected_district}")

    fig_demand = px.line(
        selected_future_df,
        x='Month',
        y='Adjusted_Predicted_Water_Demand_MLD',
        markers=True,
        labels={'Adjusted_Predicted_Water_Demand_MLD': 'Water Demand (MLD)'},
        title=f'Water Demand Prediction for {selected_district}')
    st.plotly_chart(fig_demand, use_container_width=True)

# --- End of Future Prediction Section ---

    
else:# --- Andhra Pradesh Block ---
    geojson_path = r"C:/Users/Admin/Downloads/andhra-pradesh.geojson"
    groundwater_path = r"C:/Users/Admin/Downloads/reshaped_groundwater_levels.csv"
    population_path = r"C:/Users/Admin/Downloads/andhra_pradesh_district_population.csv"
    demand_path = r"C:/Users/Admin/Downloads/andhra_pradesh_water_demand_mld.csv"

    # Load datasets
    df = pd.read_csv(groundwater_path)
    pop = pd.read_csv(population_path)
    demand = pd.read_csv(demand_path)
    gdf = gpd.read_file(geojson_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    pop.columns = pop.columns.str.strip()
    demand.columns = demand.columns.str.strip()
    gdf.columns = gdf.columns.str.strip()

    # Rename 'District' to 'district' consistently
    pop.rename(columns={'District': 'district', 'Population': 'population'}, inplace=True)
    demand.rename(columns={'District': 'district', 'Estimated_Water_Demand_MLD': 'water_demand'}, inplace=True)

    # Standardize and clean all district names
    district_rename_map = {
        "ANANTHAPURAMU": "ANANTAPUR",
        "ANANTAPURAMU": "ANANTAPUR",
        "YSR KADAPA": "YSR",
        "Y.S.R KADAPA": "YSR",
        "Y.S.R. KADAPA": "YSR",
        "SRI POTTI SRIRAMULU NELLORE": "NELLORE",
        "DR. B. R. AMBEDKAR KONASEEMA": "KONASEEMA",
        "SRI SATHYA SAI": "SATHYA SAI",
        "ALLURI SITHARAMA RAJU": "ALLURI SITARAMA RAJU",
        "BAPATLA": "BAPATLA",
        "ANNAMAYYA": "ANNAMAYYA",
        "NTR": "NTR",
        "ANAKAPALLI": "ANAKAPALLI"
    }

    for df_to_fix in [df, pop, demand]:
        df_to_fix['district'] = df_to_fix['district'].str.upper().str.strip().replace(district_rename_map)
        df_to_fix.sort_values('district', inplace=True)

    gdf['dtname'] = gdf['name'].str.upper().str.strip().replace(district_rename_map)

    # Drop rows missing groundwater levels
    df = df.dropna(subset=['groundwater_level'])

    # Merge all on 'district'
    df_latest = df.groupby('district', as_index=False)['groundwater_level'].mean()
    df_latest = df_latest.merge(pop, on='district', how='inner')
    df_latest = df_latest.merge(demand, on='district', how='inner')

    # Check if merged dataset is valid
    if df_latest.empty:
        st.error("❌ Merged data is empty. Check if district names align across files.")
        st.write("Groundwater districts:", sorted(df['district'].unique()))
        st.write("Population districts:", sorted(pop['district'].unique()))
        st.write("Demand districts:", sorted(demand['district'].unique()))
    else:
        selected_district = st.selectbox("Select District", sorted(df_latest['district'].unique()))

        # --- Groundwater Scarcity Map ---
        st.subheader("🗺️ Groundwater Scarcity Map")
        df_best = df.sort_values('groundwater_level').drop_duplicates('district')
        pred_levels = []
        for dist in df_best['district'].unique():
            dist_df = df[df['district'] == dist]
            X = pd.DataFrame({'index': range(len(dist_df))})
            y = dist_df['groundwater_level']
            if len(y) >= 2:
                model = GradientBoostingRegressor().fit(X, y)
                pred = model.predict([[len(dist_df)]])[0]
                pred_levels.append({'district': dist, 'predicted_level': pred})
        pred_df = pd.DataFrame(pred_levels)
        merged = gdf.merge(pred_df, left_on="dtname", right_on="district", how="inner")
        m = folium.Map(location=[16.5, 80.5], zoom_start=7)
        for _, row in merged.iterrows():
            level = row['predicted_level']
            status, color = classify_scarcity(level)
            folium.GeoJson(row['geometry'], style_function=lambda _, c=color: {
                "fillColor": c, "color": "black", "weight": 1, "fillOpacity": 0.7
            }, tooltip=folium.Tooltip(f"{row['dtname'].title()}: {level:.2f} m BGL ({status})")).add_to(m)
        color_legend = [
        ("Very Safe", "< 2 m", "#00FF00"),
        ("Safe", "2–5 m", "#7FFF00"),
        ("Moderate", "5–8 m", "#FFFF00"),
        ("Warning", "8–11 m", "#FFA500"),
        ("Critical", "11–15 m", "#FF4500"),
        ("Severe", "> 15 m", "#8B0000")]
        st.subheader("🖍️ Groundwater Level Color Legend")
        for label, range_text, color in color_legend:
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>⬤</span> {label}: {range_text}", unsafe_allow_html=True)
        st_folium(m, width=700)

        # --- Predicted Water Demand ---
        st.subheader("🚰 Predicted Water Demand")
        X_demand = df_latest[['population', 'groundwater_level']]
        y_demand = df_latest['water_demand']
        model = GradientBoostingRegressor().fit(X_demand, y_demand)
        df_latest['predicted_water_demand'] = model.predict(X_demand)
        if selected_district in df_latest['district'].values:
            demand_value = df_latest[df_latest['district'] == selected_district]['predicted_water_demand'].values[0]
            st.success(f"Predicted Water Demand for **{selected_district.title()}**: **{demand_value:.2f} MLD**")
        else:
            st.warning("Water demand prediction not available.")

        # --- Choropleth Map - Water Demand ---
        st.subheader("🌍 Water Demand Map")
        merged_demand = gdf.merge(df_latest, left_on="dtname", right_on="district", how="inner")
        m2 = folium.Map(location=[16.5, 80.5], zoom_start=7)
        folium.Choropleth(
            geo_data=gdf,
            data=merged_demand,
            columns=["district", "predicted_water_demand"],
            key_on="feature.properties.dtname",
            fill_color="YlGnBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Predicted Water Demand (MLD)"
        ).add_to(m2)
        for _, row in merged_demand.iterrows():
            folium.GeoJson(
                row["geometry"],
                style_function=lambda feature: {
                    "fillOpacity": 0,
                    "color": "black",
                    "weight": 0.5
                },
                tooltip=folium.Tooltip(
                    f"{row['district'].title()}<br>Demand: {row['predicted_water_demand']:.2f} MLD"
                )
            ).add_to(m2)
        st_folium(m2, width=700)
        le_ap = LabelEncoder()
        df_ap = pd.read_csv("C:/Users/Admin/Downloads/andhra_pradesh_groundwater_merged.csv")

# Standardize and clean district names
        df_ap['District'] = df_ap['district'].str.upper().str.strip()
        district_rename_map = {
            "ANANTHAPURAMU": "ANANTAPUR",
            "ANANTAPURAMU": "ANANTAPUR",
            "YSR KADAPA": "YSR",
            "Y.S.R KADAPA": "YSR",
            "Y.S.R. KADAPA": "YSR",
            "SRI POTTI SRIRAMULU NELLORE": "NELLORE",
            "DR. B. R. AMBEDKAR KONASEEMA": "KONASEEMA",
            "SRI SATHYA SAI": "SATHYA SAI",
            "ALLURI SITHARAMA RAJU": "ALLURI SITARAMA RAJU",
            "BAPATLA": "BAPATLA",
            "ANNAMAYYA": "ANNAMAYYA",
            "NTR": "NTR",
            "ANAKAPALLI": "ANAKAPALLI"}
        df_ap['District'] = df_ap['District'].replace(district_rename_map)

# Convert Month Name to Numeric
        month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        df_ap['Month_Num'] = df_ap['Month_Num'].map(month_mapping)

# Drop missing values after mapping
        required_cols = ['District', 'Month_Num', 'Rainfall_mm', 'Temperature_C','Groundwater_Level_mBGL', 'water_demand', 'Soil_Recharge_Factor', 'Population']
        df_ap = df_ap.dropna(subset=required_cols)
        df_ap['District_Code'] = le_ap.fit_transform(df_ap['District'].astype(str))

# --- Train Groundwater Level Model ---
        X_ap_gwl = df_ap[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C']]
        y_ap_gwl = df_ap['Groundwater_Level_mBGL']
        model_ap_gwl = GradientBoostingRegressor().fit(X_ap_gwl, y_ap_gwl)

# --- Train Water Demand Model ---
        X_ap_demand = df_ap[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C','Soil_Recharge_Factor', 'Groundwater_Level_mBGL']]
        y_ap_demand = df_ap['water_demand']
        model_ap_demand = GradientBoostingRegressor().fit(X_ap_demand, y_ap_demand)

# --- Forecast for Selected District ---
        selected_code = le_ap.transform([selected_district])[0]
        district_data = df_ap[df_ap['District'] == selected_district]
        if not district_data.empty:
            avg_rain = df_ap['Rainfall_mm'].mean()
            avg_temp = df_ap['Temperature_C'].mean()
            pop_growth = 0.015  # 1.5%
            base_population = district_data['Population'].iloc[0]
            soil_factor = district_data['Soil_Recharge_Factor'].iloc[0]
            future_months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            future_df = pd.DataFrame({
                'Month_Num': future_months,
                'Month': month_names,
                'District_Code': selected_code,
                'Rainfall_mm': avg_rain,
                'Temperature_C': avg_temp,
                'Soil_Recharge_Factor': soil_factor})

    # Predict groundwater levels
            X_future_gwl = future_df[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C']]
            future_df['Predicted_Groundwater_Level_mBGL'] = model_ap_gwl.predict(X_future_gwl)

    # Adjusted population growth per month
            future_df['Years_into_Future'] = future_df['Month_Num'] / 12
            future_df['Future_Population'] = base_population * ((1 + pop_growth) ** future_df['Years_into_Future'])

    # Predict water demand
            X_future_demand = future_df.copy()
            X_future_demand['Groundwater_Level_mBGL'] = future_df['Predicted_Groundwater_Level_mBGL']
            X_future_demand = X_future_demand[['District_Code', 'Month_Num', 'Rainfall_mm', 'Temperature_C','Soil_Recharge_Factor', 'Groundwater_Level_mBGL']]
            future_df['Predicted_Water_Demand_MLD'] = model_ap_demand.predict(X_future_demand)

    # Adjust water demand based on population growth
            future_df['Adjusted_Water_Demand_MLD'] = future_df['Predicted_Water_Demand_MLD'] * (future_df['Future_Population'] / base_population)

    # --- Plot Groundwater Level Forecast ---
            st.subheader(f"📉 Groundwater Level Forecast – {selected_district.title()}")
            fig_gwl = px.line(
                future_df,
                x='Month',
                y='Predicted_Groundwater_Level_mBGL',
                markers=True,
                labels={'Predicted_Groundwater_Level_mBGL': 'Groundwater Level (m BGL)'})
            st.plotly_chart(fig_gwl, use_container_width=True)

    # --- Plot Water Demand Forecast ---
            st.subheader(f"📈 Water Demand Forecast – {selected_district.title()}")
            fig_demand = px.line(
                future_df,
                x='Month',
                y='Adjusted_Water_Demand_MLD',
                markers=True,
                labels={'Adjusted_Water_Demand_MLD': 'Water Demand (MLD)'})
            st.plotly_chart(fig_demand, use_container_width=True)
        else:
            st.warning(f"No valid data found for {selected_district}")

    
    
       
