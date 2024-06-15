import streamlit as st
import requests
from datetime import datetime
from PIL import Image
from Pyspector import ZipExtractor, Inspector
import EDA 
from EDA import Analytics
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Assuming DataStorage and FeaturesGenerator classes and utils module are defined elsewhere
from Wrangling import DataStorage, FeaturesGenerator
import utils

# Define function to fetch current weather for Tallinn, Estonia
def fetch_current_weather():
    lat = 59.4370  # Latitude for Tallinn
    lon = 24.7536  # Longitude for Tallinn
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    )
    return response.json()["current_weather"]

def extract_and_process_data():
    # Data extraction
    # Example of using the class
    zip_extractor = ZipExtractor()
    zip_path = r'C:\Users\ALBER\OneDrive\Desktop\Reply Projects\data.zip'  # Replace with your ZIP file path
    password = 'your_password'  # Replace with the actual password if the ZIP file is encrypted, or set to None
    batch_needed = zip_extractor.extract_zip_file(zip_path, password)

    data_storage = DataStorage()
    features_generator = FeaturesGenerator(data_storage=data_storage)

    df_train_features = features_generator.generate_features(data_storage.df_data)
    df_train_features = df_train_features[df_train_features['target'].notnull()]
    df_train_features = utils.clean_data(df_train_features , column_threshold=0.6)
    # df_train_features.isnull().sum().sum()
    df_train_features.info()

    return df_train_features , data_storage

def generate_report():
    pyspector = Inspector(r'C:\Users\ALBER\OneDrive\Desktop\Reply Projects\data', "Enefit Dataset Inspection", "Consumers Prosumer Analysis", "Generated on: " + datetime.now().strftime('%Y-%m-%d') + 
                                    " This report provides a detailed analysis of the datasets in order to evaluate data quality and identify potential issues.")
    pyspector.generate_report()


def main():
    st.set_page_config(page_title="Estonia Energy Prosumer Analysis", layout="wide")
    st.title("Energy Prosumer Analysis Platform")
    image = Image.open('image.png')
    st.image(image, caption='Enefit Energy Analysis', use_column_width=False)

    weather = fetch_current_weather()
    st.subheader("Current Weather in Tallinn, Estonia")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{weather['temperature']}°C")
    col2.metric("Wind Speed", f"{weather['windspeed']} km/h")
    col3.metric("Wind Direction", f"{weather['winddirection']}°")

    if st.button('Load and Process Data'):
        df_train_features, data_storage = extract_and_process_data()
        st.session_state['data_storage'] = data_storage
        st.session_state['df_train_features'] = df_train_features
        st.success('Data loaded and processed successfully!')
        st.write(df_train_features.info())
        st.write("Total missing values:", df_train_features.isnull().sum().sum())

    if st.button('Generate Report'):
        report_path = generate_report()
        st.success(f'Report generated: {report_path}')
        st.download_button("Download Report", open(r'C:\Users\ALBER\OneDrive\Desktop\Reply Projects\data\data_inspection_report.pdf', "rb"), file_name="report.pdf")

    if st.button('Add Data'):
        st.write("Add Data functionality to be implemented.")

    # Navigation and pages shown after data is loaded
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        page = st.sidebar.radio("Go to", ("Analytics", "Consumer Segmentation", "Prosumer Segmentation", "Power Grid Management"))
        display_analytics_content(page)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ("Project Structure & Backend", "Analytics"))

    # Display appropriate page content based on sidebar navigation choice
    if page == "Project Structure & Backend":
        display_project_structure_content()
    elif page == "Analytics":
        display_analytics_content()


    st.markdown("### About Enefit")
    st.text("As a leading energy company in the Baltic region, Enefit is committed to using environmentally friendly energy solutions to help customers plan and implement their green journeys.")

def display_project_structure_content():
    st.subheader("Project Structure & Backend")
    st.write("Details about the project structure and backend functionalities will be provided here.")

    # Assuming the image is saved in the same directory as your script or specify the path.
    image_path = 'workflow.png'  # Update the path if your image is in a different directory.

    # Display the image with a caption
    st.image("workflow.png", caption='Visual Representation of the Project Workflow')
    st.write("""
Objectives:
Data Segmentation: To categorize customers into meaningful segments for targeted marketing and personalized customer service.
Feature Analysis: To identify key features that significantly impact customer behavior and segmentation.
Model Optimization: To develop and refine predictive models that can accurately classify customer types and predict their future actions.
Workflow Details:
Data Ingestion:

The process begins with the ingestion of raw customer data, setting the stage for further manipulation and analysis.
Data Wrangling:

This step addresses data cleaning and transformation tasks to rectify inconsistencies, handle missing values, and prepare the dataset for analysis.
Data Preprocessing:

Standardization and normalization techniques are applied to ensure that the data is suitable for machine learning algorithms, which require a standardized input format to perform optimally.
Clustering Method Selection:

A decision point where the best clustering method is chosen based on the data characteristics. Options include KMeans, Hierarchical, and KMedoids clustering, each suitable for different types of data distributions and segmentation needs.
Clustering Execution and Evaluation:

The selected clustering algorithm is applied to categorize customers into segments. The effectiveness of the segmentation is evaluated using metrics like the silhouette score.
Classifier Selection and Training:

Another decision point leads to the selection of a machine learning classifier such as Random Forest, XGBoost, CatBoost, or LightGBM. The classifier is trained on labeled data derived from the clustering output.
Model Evaluation and Optimization:

Models are rigorously evaluated using metrics like accuracy, precision, recall, and F1 score. The best performing model is optimized through techniques like hyperparameter tuning and cross-validation.
Customer Segmentation:

The finalized model is used to assign new customers to existing segments, ensuring that each customer is categorized according to the most recent data-driven insights.
Analysis of Important Features:

Key features influencing customer segmentation are identified, providing insights into which attributes most significantly affect customer behavior.
Customer Segment Impact Analysis:

The impact of different customer segments on targeted business outcomes (like sales, customer retention) is analyzed, offering actionable insights for strategic decision-making.
Conclusion:
The project harnesses advanced data processing techniques and machine learning to deeply understand customer behaviors and preferences. The insights garnered from this analysis not only enhance customer experience but also boost business efficiency by enabling more informed decision-making. This workflow is adaptable and can be extended or modified to meet specific business needs or to handle different types of customer data.
""")
   
    
def display_analytics_content():
    st.subheader("Analytics")
    analytics_page = st.selectbox("Select Analytics Section:", ("EDA", "Consumer Segmentation", "Prosumer Segmentation", "Power Grid Management"))
    if analytics_page == "EDA":
        display_eda_content()
    elif analytics_page == "Consumer Segmentation":
        display_consumer_segmentation_content()
    elif analytics_page == "Prosumer Segmentation":
        display_prosumer_segmentation_content()
    elif analytics_page == "Power Grid Management":
        display_power_grid_management_content()

def display_eda_content():
    st.subheader("Exploratory Data Analysis (EDA)")
    if 'data_storage' not in st.session_state:
        st.warning("Data not loaded. Please load data first.")
        return

    data_storage = st.session_state['data_storage']  # Access from session state

    # Dataset options as dictionary
    dataset_options = {
        "Client Data": data_storage.df_client,
        "Training Data": data_storage.df_data,
        "Electricity Prices": data_storage.df_electricity_prices,
        "Forecast Weather": data_storage.df_forecast_weather,
        "Historical Weather": data_storage.df_historical_weather,
        "Overall": None  # Option for overall profiling
    }

    dataset_name = st.selectbox("Select Dataset for Profiling:", list(dataset_options.keys()))

    if dataset_name != "Overall":
        selected_df = pd.DataFrame(dataset_options[dataset_name])
        profile = ProfileReport(selected_df, explorative=True, title=f"Profile of {dataset_name}")
        st_profile = st.components.v1.html(profile.to_html(), height=800, width=800, scrolling=True)

        if st.button(f"Download {dataset_name} Profile Report"):
            profile.to_file(f"{dataset_name}_profile_report.html")
            with open(f"{dataset_name}_profile_report.html", "rb") as file:
                st.download_button(label=f"The file {dataset_name} is ready to download", data=file, file_name=f"{dataset_name}_profile_report.html")



    else:
        data_storage = st.session_state['data_storage']  # Access from session state
        # Custom EDA for 'Overall' selection
        cat_columns = ['county', 'is_business', 'product_type']
        num_columns = ['eic_count', 'installed_capacity']
        st.set_option('deprecation.showPyplotGlobalUse', False)
        eda = EDA.Exploratory_analysis(data_storage.df_client, cat_columns, num_columns)
        st.pyplot(eda.plot_stripplots())
        st.pyplot(eda.plot_boxplots())

import streamlit as st
import os
import pandas as pd
from Costumer_segment import SegmentationViz, CustomerSegmentation

def display_prosumer_segmentation_content():
    st.subheader("Prosumer Segmentation")

    # Check if the data is loaded
    if 'df_train_features' not in st.session_state:
        st.error("Data not loaded. Please load data first.")
        return

    # Define columns for segmentation
    columns_seg = [
        "target", "county", "is_business", "product_type", "is_consumption",
        "eic_count", "installed_capacity", "10_metre_u_wind_component",
        "10_metre_v_wind_component", "direct_solar_radiation",
        "surface_solar_radiation_downwards", "snowfall", "total_precipitation",
        "rain", "surface_pressure", "windspeed_10m", "winddirection_10m",
        "shortwave_radiation", "diffuse_radiation", "cloudcover_high",
        "cloudcover_low", "cloudcover_mid"
    ]
    df_train_features_fil = st.session_state['df_train_features'][columns_seg]
    prosumers_df = df_train_features_fil[df_train_features_fil["is_consumption"] == 0]

    # Input box for selecting the number of clusters
    num_clusters = st.number_input('Enter number of clusters for analysis:', min_value=2, max_value=10, value=3, step=1)
    st.session_state['num_clusters']=num_clusters
    column_input = st.text_input("Enter the column names for analysis, separated by commas (e.g., total_precipitation, shortwave_radiation):")
    columns_to_evaluate = [x.strip() for x in column_input.split(',')] if column_input else []

    # Button to perform cluster analysis
    if st.button('Run Cluster Analysis'):
        st.session_state['segmentation_results_prosumers'] = SegmentationViz(prosumers_df.iloc[:20000], "target", n_clusters=num_clusters)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(st.session_state['segmentation_results_prosumers'].run_analysis())
    
    # Button to perform segmentation
    if st.button('Run Segmentation'):
        if 'segmentation_results' not in st.session_state:
            st.error("Please run the cluster analysis first.")
            return
        st.session_state['customer_segment_prosumers'] = CustomerSegmentation(prosumers_df.iloc[:10000], "target", num_clusters)
        st.pyplot(st.session_state['customer_segment_prosumers'].run_pipeline('hierarchical'))
        st.session_state['prosumers_df_segmented_prosumers'] = st.session_state['customer_segment_prosumers'].create_customer_types_for_data(prosumers_df)
        df_train_features = st.session_state['df_train_features']
        prosumers_df = df_train_features[df_train_features_fil["is_consumption"] == 0]
        

        
    if st.button('Analysis of important Features'):
            # Allow user to input column names


        df_train_features = st.session_state['df_train_features']
        prosumers_df = df_train_features[df_train_features_fil["is_consumption"] == 0]
        # Example usage:
        customer_segment = CustomerSegmentation(prosumers_df.iloc[:10000], 'target', num_clusters)
        customer_segment.run_pipeline('hierarchical')
        prosumers_df = customer_segment.create_customer_types_for_data(prosumers_df)
        analytics = Analytics(prosumers_df)
        cumulative_data_prosumers = analytics.plot_cumulative_target_across_months()
        grouped_data_daily_prosumers = analytics.summarize_target_by_time_unit('daily')
        st.session_state['grouped_data_daily_prosumers'] = grouped_data_daily_prosumers
        # boxplot_columns = ["total_precipitation", "shortwave_radiation", "surface_solar_radiation_downwards"]
        st.pyplot(analytics.plot_all_in_one_figure(cumulative_data_prosumers, grouped_data_daily_prosumers, columns_to_evaluate))
        



    # Button to save row IDs
    if st.button('Save Row IDs'):
        if 'prosumers_df_segmented' not in st.session_state:
            st.error("Please perform the segmentation first.")
            return
        prosumers_df_segmented = st.session_state['prosumers_df_segmented']
        customer_types = prosumers_df_segmented['customer_type'].unique()
        dataframes = {ctype: prosumers_df_segmented[prosumers_df_segmented['customer_type'] == ctype] for ctype in customer_types}

        # Save and provide download buttons for each customer type
        for ctype, df in dataframes.items():
            filename = f"cluster_{ctype}_row_ids.txt"
            filepath = os.path.join(r'C:\Users\ALBER\OneDrive\Desktop\Reply Projects', filename)
            with open(filepath, 'w') as file:
                file.write('\n'.join(map(str, df.index.tolist())))  # Convert index to string and write
            st.write(f"Cluster {ctype} - {len(df)} records")
            with open(filepath, "rb") as file:
                st.download_button(label=f"Download Cluster {ctype} row IDs",
                                   data=file,
                                   file_name=filename,
                                   mime='text/plain')






def display_consumer_segmentation_content():
    st.subheader("Consumer Segmentation")

    # Check if the data is loaded
    if 'df_train_features' not in st.session_state:
        st.error("Data not loaded. Please load data first.")
        return

    # Define columns for segmentation
    columns_seg = [
        "target", "county", "is_business", "product_type", "is_consumption",
        "eic_count", "installed_capacity", "10_metre_u_wind_component",
        "10_metre_v_wind_component", "direct_solar_radiation",
        "surface_solar_radiation_downwards", "snowfall", "total_precipitation",
        "rain", "surface_pressure", "windspeed_10m", "winddirection_10m",
        "shortwave_radiation", "diffuse_radiation", "cloudcover_high",
        "cloudcover_low", "cloudcover_mid"
    ]
    df_train_features_fil = st.session_state['df_train_features'][columns_seg]
    consumers_df = df_train_features_fil[df_train_features_fil["is_consumption"] == 1]

    # Input box for selecting the number of clusters
    num_clusters = st.number_input('Enter number of clusters for analysis:', min_value=2, max_value=10, value=3, step=1)
    column_input = st.text_input("Enter the column names for analysis, separated by commas (e.g., total_precipitation, shortwave_radiation):")
    columns_to_evaluate = [x.strip() for x in column_input.split(',')] if column_input else []

    # Button to perform cluster analysis
    if st.button('Run Cluster Analysis'):
        st.session_state['segmentation_results'] = SegmentationViz(consumers_df.iloc[:20000], "target", n_clusters=num_clusters)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(st.session_state['segmentation_results'].run_analysis())
    
    # Button to perform segmentation
    if st.button('Run Segmentation'):
        if 'segmentation_results' not in st.session_state:
            st.error("Please run the cluster analysis first.")
            return
        st.session_state['customer_segment'] = CustomerSegmentation(consumers_df.iloc[:10000], "target", num_clusters)
        st.pyplot(st.session_state['customer_segment'].run_pipeline('hierarchical'))
        st.session_state['conosumers_df_segmented'] = st.session_state['customer_segment'].create_customer_types_for_data(consumers_df)
        df_train_features = st.session_state['df_train_features']
        consumers_df = df_train_features[df_train_features_fil["is_consumption"] == 1]
        

        
    if st.button('Analysis of important Features'):
            # Allow user to input column names


        df_train_features = st.session_state['df_train_features']
        consumers_df = df_train_features[df_train_features_fil["is_consumption"] == 1]
        # Example usage:
        customer_segment = CustomerSegmentation(consumers_df.iloc[:10000], 'target', num_clusters)
        customer_segment.run_pipeline('hierarchical')
        consumers_df = customer_segment.create_customer_types_for_data(consumers_df)
        analytics = Analytics(consumers_df)
        cumulative_data_consumers = analytics.plot_cumulative_target_across_months()
        grouped_data_daily_consumers = analytics.summarize_target_by_time_unit('daily')
        st.session_state['grouped_data_daily_consumers'] = grouped_data_daily_consumers
        # boxplot_columns = ["total_precipitation", "shortwave_radiation", "surface_solar_radiation_downwards"]
        st.pyplot(analytics.plot_all_in_one_figure(cumulative_data_consumers, grouped_data_daily_consumers, columns_to_evaluate))
        



    # Button to save row IDs
    if st.button('Save Row IDs'):
        if 'consumers_df_segmented' not in st.session_state:
            st.error("Please perform the segmentation first.")
            return
        consumers_df_segmented = st.session_state['consumers_df_segmented']
        customer_types = consumers_df_segmented['customer_type'].unique()
        dataframes = {ctype: consumers_df_segmented[consumers_df_segmented['customer_type'] == ctype] for ctype in customer_types}

        # Save and provide download buttons for each customer type
        for ctype, df in dataframes.items():
            filename = f"cluster_{ctype}_row_ids.txt"
            filepath = os.path.join(r'C:\Users\ALBER\OneDrive\Desktop\Reply Projects', filename)
            with open(filepath, 'w') as file:
                file.write('\n'.join(map(str, df.index.tolist())))  # Convert index to string and write
            st.write(f"Cluster {ctype} - {len(df)} records")
            with open(filepath, "rb") as file:
                st.download_button(label=f"Download Cluster {ctype} row IDs",
                                   data=file,
                                   file_name=filename,
                                   mime='text/plain')



def display_power_grid_management_content():
    # Check if the number of clusters is set in the session state; if not, set it using a number input
    if 'num_clusters' not in st.session_state:
        st.session_state['num_clusters'] = st.number_input('Enter number of clusters for analysis:', min_value=2, max_value=10, value=3, step=1)

    # Check if required data is loaded in the session state
    if 'grouped_data_daily_prosumers' not in st.session_state or 'grouped_data_daily_consumers' not in st.session_state:
        st.error("Please ensure all required analyses are run before accessing the Power Grid Management page.")
        return

    # Retrieve data from the session state
    grouped_data_daily_prosumers = st.session_state['grouped_data_daily_prosumers']
    grouped_data_daily_consumers = st.session_state['grouped_data_daily_consumers']

    # Display sliders that are controlled by the number of clusters stored in the session state
    num_clusters_consumers = st.slider('Select number of clusters for consumers', min_value=0, max_value=st.session_state['num_clusters']-2, value=st.session_state['num_clusters']-2)
    num_clusters_prosumers = st.slider('Select number of clusters for prosumers', min_value=0, max_value=st.session_state['num_clusters']-2, value=st.session_state['num_clusters']-2)

    # Display the plot with the selected number of clusters
    st.pyplot(EDA.plot_consumer_prosumer_data(
        grouped_data_daily_consumers,
        grouped_data_daily_prosumers,
        num_clusters_consumers,
        num_clusters_prosumers
    ))

if __name__ == "__main__":
    main()
