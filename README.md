# apartment-price-ai

This project aims to predict the price of an apartment based on various features like district, square footage, floor number, number of rooms, year of construction, and building type. The model helps users determine approximate apartment prices using machine learning algorithms.

## Project Description

This AI model leverages different machine learning techniques, such as Linear Regression, Random Forest, and K-means clustering, to predict apartment prices based on user-provided parameters. The project uses real estate data collected from Krisha.kz, a popular real estate listing platform in Kazakhstan. The goal is to help buyers and renters make more informed decisions when evaluating property listings.

### Features:
- **Price Prediction**: Predict apartment prices based on characteristics such as district, square footage, floor number, and number of rooms.
- **K-means Clustering**: Classify apartments into different clusters based on area and the number of rooms.
- **Association Rules**: Use the Apriori algorithm to discover association rules within real estate listings.
- **PCA**: Perform Principal Component Analysis to reduce data dimensionality and better understand key features.

## Installation

1. Clone the repository to your local machine (or download the ZIP).

    ```bash
    git clone https://github.com/moonymoonymoo/apartment_price_ai.git
    ```

2. Navigate to the project directory.

    ```bash
    cd apartment_price_ai
    ```

3. Install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have **Python 3** installed on your system along with necessary libraries.

## Usage

1. Run the project using Streamlit to launch the web application.

    ```bash
    streamlit run app.py
    ```

2. Enter the required details of the apartment (district, square footage, floor number, rooms, year of construction, building type) and select the desired machine learning model (Linear Regression, K-means, PCA, etc.) to get the predicted price.

3. View results:
    - The model will output a predicted price based on your input.
    - You can also view clustering results and association rules using the other models.

## Files:

- **app.py**: Main application file that runs the web interface.
- **apriori_algorithm.py**: Implements the Apriori algorithm for association rule mining.
- **pca_algorithm.py**: Implements the Principal Component Analysis (PCA).
- **data.csv**: Dataset containing real estate listings for training the models.
- **test_sift.py**: Script to test SIFT for feature matching (optional for image processing tasks).
- **README.md**: Project documentation.

## Algorithms Used:

1. **Linear Regression**: Predicts the apartment price based on the relationship between various features and the price.
2. **Random Forest**: An ensemble learning method used to predict the price by combining several decision trees.
3. **K-means Clustering**: Used to group similar apartments based on square footage and the number of rooms.
4. **Apriori Algorithm**: Identifies frequent itemsets in real estate listings and generates association rules.
5. **PCA (Principal Component Analysis)**: Reduces data dimensionality for better understanding and visualization of key features.

## Contributing

Feel free to fork this project, open issues, and submit pull requests if you would like to contribute.

## License

This project is open source and available under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Created by [moonymoonymoo](https://github.com/moonymoonymoo)  
Date: 12 May 2025
