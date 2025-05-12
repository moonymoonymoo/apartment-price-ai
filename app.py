import streamlit as st 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.cluster import KMeans

# Import our algorithms
from apriori_pca_project.apriori_algorithm import apriori_algorithm  # Import Apriori algorithm
from apriori_pca_project.pca_algorithm import pca_algorithm  # Import PCA algorithm

# Page setup
st.set_page_config(page_title="🏠 Apartment AI", layout="centered")
st.title("🏢 AI Apartment Price Prediction")

st.markdown("Fill in the apartment details below to get an approximate price 💰")

# --- User input fields ---
district = st.selectbox("District", ["Almaly", "Alatau", "Auezov", "Bostandyk", "Zhetysu"])
square = st.slider("Apartment Area (m²)", 30, 150, 60)
floor = st.number_input("Floor", min_value=1, max_value=25, value=5)
rooms = st.selectbox("Number of Rooms", [1, 2, 3, 4, 5])
year = st.slider("Year of Construction", 1980, 2023, 2010)
building_type = st.selectbox("Building Type", ["Brick", "Panel", "Monolith", "Block"])

# --- Select model for prediction ---
selected_algorithm = st.selectbox("Select algorithm for clustering", 
                                 ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", 
                                  "Naive Bayes", "KNN", "SVM", "Gradient Boosting", "Apriori", "PCA", "K-means"])

# --- Predict apartment price ---
if st.button("📊 Predict Price"):
    # Collect data from form
    user_input = pd.DataFrame([{
        "district": district,
        "square": square,
        "floor": floor,
        "rooms": rooms,
        "year": year,
        "building_type": building_type
    }])

    # Encode categorical data
    df_encoded = user_input.copy()
    for col in ["district", "building_type"]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Train models
    df = pd.read_csv("data.csv")
    for col in ["district", "building_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsRegressor(),
        "SVM": SVR(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            prediction = model.predict(df_encoded)[0]
            results[name] = f"${prediction:,.0f}"
        except Exception as e:
            results[name] = f"Error: {e}"

    st.subheader("💡 Model Predictions:")
    st.table(pd.DataFrame(results.items(), columns=["Model", "Predicted Price"]).reset_index(drop=True).set_index(pd.Index(range(1, len(results) + 1))))

    # --- Algorithm selection and execution ---
    if selected_algorithm == "K-means":
        # K-means clustering code
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['square', 'rooms']])

        # Create plot (based on area and number of rooms)
        fig = px.scatter(df, x="square", y="rooms", color="cluster", 
                         labels={"square": "Area (m²)", "rooms": "Number of Rooms"},
                         title="Clustering Apartments by Area and Number of Rooms",
                         color_continuous_scale='viridis')

        # Highlight the selected apartment
        user_input['cluster'] = kmeans.predict(df_encoded[['square', 'rooms']])
        fig.add_scatter(x=user_input['square'], y=user_input['rooms'], mode='markers', marker=dict(color='red', size=12), name='Selected Apartment')

        # Display the plot on the site
        st.plotly_chart(fig)

    elif selected_algorithm == "Apriori":
        # Run Apriori algorithm
        rules = apriori_algorithm(df)  # Call the Apriori function
        st.subheader("Apriori Association Rules:")
        st.write(rules)  # Display the rules in a table

    elif selected_algorithm == "PCA":
        # Run PCA algorithm
        pca_result = pca_algorithm(df)  # Call the PCA function
        st.write(pca_result)  # Display the PCA result
