# E-Commerce Marketing

Optimize the marketing budget to enhance revenue response by analyzing the sales trends across different product sub-categories.

- Focus Areas:
  - Camera Accessories
  - Home Audio
  - Gaming Accessories

## Dataset

The project uses the "ConsumerElectronics.csv" dataset, which contains historical sales data for various consumer electronics products. The dataset includes information about order date, product details, units sold, GMV, and other relevant features.

## Methodology

1. **Data Cleaning and Preprocessing**: The initial step involves cleaning the dataset by handling missing values, converting data types, and removing duplicates.

2. **Exploratory Data Analysis (EDA)**: EDA is performed to understand the data distribution, identify trends and patterns, and gain insights into the relationships between variables. This step includes visualizing data using histograms, scatter plots, and time series plots.

3. **Feature Engineering**: New features are created to enhance the model's predictive power. These features include:
    - Aggregated GMV and units sold per week for each sub-category as per business requirement.
    - **Pay Date Flag**: Indicates if the week contains typical pay dates (1st or 15th of the month).
    - **Holiday Indicator**: Marks significant Canadian holidays (Christmas, New Year, Canada Day).
    - **Climate Data Integration**: Includes average weekly temperature data for Ontario.
    - **Average Order Value (AOV)**: Calculated as Total GMV / Total Units.

4. **Model Building**: Two approaches are used for model building:
    - **Machine Learning**: Linear Regression, XGBRegressor, RandomForestRegressor, and Support Vector Machine (SVM) models are trained and evaluated.
    - **Deep Learning**: A sequential model with dense layers and dropout is implemented using TensorFlow and Keras.

5. **Model Evaluation**: Models are evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score. Permutation feature importance is also analyzed to understand the impact of different features on model predictions.

## Results

- The machine learning models show promising results in predicting weekly GMV for the chosen sub-categories.
- The deep learning model exhibits similar performance and potentially captures more complex relationships.
- Feature importance analysis reveals that `total_units` and `Average_Order_Value` are the most significant predictors of sales.
- The importance of product subcategories `Gaming Accessory` and `Home Audio` have some importance, but they are much lower than total_units. These represent the
  sub-categories of products and seem to have some impact.

## Conclusion

This project provides valuable insights into e-commerce sales trends and demonstrates the effectiveness of using machine learning and deep learning for sales prediction. The models can be used to forecast future sales, optimize inventory management, and inform marketing strategies.

## Recommendation to Re-allocate Marketing Budget:
- Focus more on Camera Accessories and Home Audio during September, as they show peak sales during that period.
- Optimize discounts for Gaming Accessories to capitalize on their price-sensitive customers.
- Review and refine ad campaigns to ensure investments yield expected returns

### Streamline Delivery Logistics:
- Although delivery speed doesn’t significantly impact sales, it’s still valuable for customer satisfaction—focus on maintaining SLA consistency.

## Further Work
 
- Experiment with different model architectures and hyperparameters.
- Deploy the models for real-time sales forecasting and decision-making.
