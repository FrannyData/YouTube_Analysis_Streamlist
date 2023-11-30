import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

def load_config() -> dict:
    """
    Load the configuration file. This will be a json file called config.json.
    
    Returns:
        dict: Dictionary containing the configuration parameters.
    """
        
    with open('config.json') as f:
        return json.load(f)



# Load Aggregated Metrics By Video
def load_df_agg() -> pd.DataFrame:
    """
    Load and process the Aggregated Metrics By Video dataset.
    
    Returns:
        pd.DataFrame: Processed dataframe with columns like 'Video', 'Video Title', 'Video publish time', and so on.
    """
    
    df = pd.read_csv(config['paths']['df_agg']).iloc[1:,:]
    
    df.columns = ['Video', 'Video Title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes', 
                  'Subscribers lost', 'Subscribers gained', 'RPM (USD)', 'CPM (USD)', 'Average percentage viewed (%)', 
                  'Average view duration', 'Views', 'Watch time (hours)', 'Subscribers', 'Your estimated revenue (USD)', 
                  'Impressions', 'Impressions click-through rate (%)']
    
    df['Video publish time'] = pd.to_datetime(df['Video publish time'], format='%b %d, %Y')
    df['Average view duration'] = df['Average view duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    df['Avg_duration_sec'] = df['Average view duration'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second)
    df['Engagement_ratio'] = (df['Comments added'] + df['Shares'] + df['Dislikes'] + df['Likes']) / df['Impressions']
    df['Views / sub gained'] = df['Views'] / df['Subscribers gained']
    df.sort_values(by='Video publish time', inplace=True, ascending=False)
    
    return df

# Load Aggregated Metrics By Country And Subscriber Status
def load_df_agg_sub() -> pd.DataFrame:
    """
    Load the Aggregated Metrics By Country And Subscriber Status dataset.
    
    Returns:
        pd.DataFrame: Dataframe containing the metrics by country and subscriber status.
    """

    return pd.read_csv(config['paths']['df_agg_sub'])


# Load Aggregated Metrics By Video (for comments)
def load_df_comments() -> pd.DataFrame:
    """
    Load the Aggregated Metrics By Video dataset for comments.
    
    Returns:
        pd.DataFrame: Dataframe containing the metrics by video.
    """
    
    return pd.read_csv(config['paths']['df_comments'])


# Load Video Performance Over Time
def load_df_time() -> pd.DataFrame:
    """
    Load and process the Video Performance Over Time dataset.
    
    Returns:
        pd.DataFrame: Processed dataframe with metrics over time.
    """
    
    df = pd.read_csv(config['paths']['df_time'])
    df['Date'] = df['Date'].str.replace('Sept', 'Sep')
    df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
    
    return df


# Main function to load all data
def load_data() -> tuple:
    """
    Load all datasets and return them as a tuple.
    
    Returns:
        tuple: Tuple containing four dataframes - df_agg, df_agg_sub, df_comments, df_time.
    """
    
    df_agg = load_df_agg()
    df_agg_sub = load_df_agg_sub()
    df_comments = load_df_comments()
    df_time = load_df_time()
    
    return df_agg, df_agg_sub, df_comments, df_time



# Defining style functions
def style_negative_values(v, props=''):
    """Style negative values in dataframe"""
    
    try:
        return props if v < 0 else None
    except:
        pass
   
def style_positive_values(v, props=''):
    """Style positive values in dataframe"""
    
    try:
        return props if v > 0 else None
    except:
        pass

def audience_simple(country):
    """Simplifica el código de país a los tres principales."""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'




def process_data(df_agg, df_time):
    #create dataframes from the function 

    #additional data engineering for aggregated data 
    df_agg_diff = df_agg.copy()
    metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12) #this line creates a variable that is the max date minus 12 months

    numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
    median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo].iloc[:, numeric_cols].median()

    #create differences from the median for values 
    #Just numeric columns 
    numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
    df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)


    #merge daily data with publish data to get delta 
    df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
    df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

    # get last 12 months of data rather than all data 
    date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
    df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

    # get daily view data (first 30), median & percentiles 
    views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
    views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
    views_days = views_days[views_days['days_published'].between(0,30)]
    views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']] 
    views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()

    return df_time_diff, df_agg_diff, views_cumulative




def main():
    # Create dataframes for each tab from the function above
    df_agg, df_agg_sub, df_comments, df_time = load_data()
    
    df_time_diff, df_agg_diff, views_cumulative = process_data(df_agg, df_time)

    #-----------------------------------------
    #******** Building Streamlit App *********
    #-----------------------------------------

    #analysis type is a sidebar selection
    analysis_type = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics','Individual Video Analysis'))


    #Show individual metrics 
    if analysis_type == 'Aggregate Metrics':
        st.write("YouTube Aggregated Data")
        
        df_agg_metrics = df_agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM (USD)', 'Average percentage viewed (%)', 
                             'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']] 
        metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6) #this line creates a variable that is the max date minus 6 months, because we want to look at the last 6 months
        metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12) #this line creates a variable that is the max date minus 12 months
        metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median() #this line creates a variable that is the median of the numeric columns for the last 6 months
        metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median()
        
        # Exclude date columns from metric_medians6mo and metric_medians12mo before the loop
        metric_medians6mo = metric_medians6mo.drop(['Video publish time'], errors='ignore')
        metric_medians12mo = metric_medians12mo.drop(['Video publish time'], errors='ignore')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
        
        count = 0
        for i in metric_medians6mo.index:
            with columns[count]:
                delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
                st.metric(label=i, value = round(metric_medians6mo[i], 1), delta = "{:.2%}".format(delta))
                count += 1 
                if count >= 5:
                    count = 0
        #get date information / trim to relevant data 
        df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
        df_agg_diff_final = df_agg_diff.loc[:,['Video Title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM (USD)','Average percentage viewed (%)',
                                'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
        
        
        numeric_cols = (df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64')
    
        #now we will create a list of the columns in the dataframe
        df_agg_numeric_lst = df_agg_diff.columns[numeric_cols].tolist()
        
        df_to_pct = {}
        for i in df_agg_numeric_lst:
            df_to_pct[i] = '{:.1%}'.format
            
        st.dataframe(df_agg_diff_final.style.hide().applymap(style_negative_values, props='color:red').applymap(style_positive_values, props='color:green').format(df_to_pct))
        
    if analysis_type == 'Individual Video Analysis':
        videos = tuple(df_agg['Video Title'])
        st.write("Individual Video Performance")
        video_select = st.selectbox('Pick a Video:', videos)
        
        agg_filtered = df_agg[df_agg['Video Title'] == video_select]
        agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
        agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
        agg_sub_filtered.sort_values('Is Subscribed', inplace= True)   
        
        fig = px.bar(agg_sub_filtered, x ='Views', y='Is Subscribed', color ='Country', orientation ='h')
        #order axis 
        st.plotly_chart(fig)
        
        agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
        first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
        first_30 = first_30.sort_values('days_published')
        
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                        mode='lines',
                        name='20th percentile', line=dict(color='purple', dash ='dash')))
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                            mode='lines',
                            name='50th percentile', line=dict(color='black', dash ='dash')))
        fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                            mode='lines', 
                            name='80th percentile', line=dict(color='royalblue', dash ='dash')))
        fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                            mode='lines', 
                            name='Current Video' ,line=dict(color='firebrick',width=8)))
            
        fig2.update_layout(title='View comparison first 30 days',
                    xaxis_title='Days Since Published',
                    yaxis_title='Cumulative views')
        
        st.plotly_chart(fig2)
        


# Load config file
config = load_config()


# Run main function        
if __name__ == "__main__":
    main()