# YouTube Data Analysis Dashboard with Streamlit and Python

This project creates a YouTube data analysis dashboard using Streamlit, a web application framework, in Python. It's designed for aggregating and visualizing YouTube video metrics to aid content creators and marketers.

## Libraries
* **Pandas**: Used for data manipulation and analysis.
* **NumPy**: Adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
* **Plotly**: A graphing library makes interactive, publication-quality graphs online.
* **Streamlit**: Turns data scripts into shareable web apps.
* **Datetime**: Supplies classes for manipulating dates and times.


## Data Loading and Processing

### 1. Configuration and Dataset Loading
* Configurations are loaded from a JSON file.
* Four main datasets are loaded: Aggregated Metrics By Video, By Country And Subscriber Status, comments, and Video Performance Over Time.

### 2. Data Processing Functions
* These functions process CSV files into Pandas DataFrames, applying transformations like datetime parsing and calculation of new metrics (e.g., Engagement Ratio, Average Duration in Seconds).

## Data Visualization and Analysis
### 1. Building Streamlit App
* Streamlit is used to create an interactive sidebar for analysis type selection.

### 2. Aggregate Metrics Analysis
* This section of the code generates an interactive dashboard for aggregated YouTube data analysis.

### 3. Individual Video Analysis
* Here, detailed performance metrics of individual YouTube videos are presented.


## Styling and Presentation
### 1. Streamlit Styling and Layouts
* Streamlit's column layout is utilized for arranging metrics on the dashboard.

### 2. Data Styling Functions
* These functions are used to apply conditional styling to the data presented on the dashboard, enhancing its readability and visual appeal.



## Conclusion
This dashboard is a highly versatile tool for YouTube analytics. It combines data processing with interactive visualizations, providing relevant metrics on video performance and audience engagement. The use of Streamlit makes the dashboard accessible and easy to use, ideal for both beginners and advanced users.





[Wiki](https://github.com/FrannyData/YouTube_Analysis_Streamlist/wiki)

