import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime,timedelta,date

st.set_page_config(
    page_title="Labor Planning Model",
    page_icon=":brain:",
    layout="wide"
)

st.title("Labor Planning Model")
st.write("This is a prototype tool in development to compute labor planning requirement based on input parameters")

forecast_template = pd.read_csv("input_files/forecast_input_file.csv")
forecast_template_str = forecast_template.to_csv(index=False)  


productivity_template = pd.read_csv("input_files/process_rate_template.csv")  
productivity_template_str = productivity_template.to_csv(index=False)

# data input

shift_hrs=st.text_input('Shift hours per day','8')

st.sidebar.header("Please input your data here")
st.sidebar.subheader("Forecasts")
st.sidebar.download_button(label="Click to download a forecast template",data=forecast_template_str,file_name='forecast_template.csv',mime='text/csv')
forecast_file=st.sidebar.file_uploader("Upload forecast file")

st.sidebar.subheader("Process rates")
st.sidebar.download_button(label="Click to download a rate template file",data=productivity_template_str,file_name='process_rate_template.csv',mime='text/csv')
rate_file=st.sidebar.file_uploader("Upload process rate file")

if forecast_file is None:
    st.info(" Upload a forecast file through config")
    st.stop()

if rate_file is None:
    st.info(" Upload a process rate file through config")
    st.stop()


df_forecast=pd.read_csv(forecast_file,skiprows=1,names=['business_unit1','business_unit2','date','process_1','forecast'])
df_rate=pd.read_csv(rate_file,skiprows=1,names=['business_unit1','business_unit2','process_1','process_2','unit_rate'])

#df_forecast=pd.read_csv("r'C:\Users\mistryn\Documents\streamlit_laborplan-main\input_files\\"forecast_input_file.csv',skiprows=1,names=['business_unit1','business_unit2','date','process_1','forecast'])
#df_rate=pd.read_csv(r'C:\Users\mistryn\Documents\streamlit_laborplan-main\input_files\\process_rate_template.csv',skiprows=1,names=['business_unit1','business_unit2','process_1','process_2','unit_rate'])


df_forecast['date']=pd.to_datetime(df_forecast['date']).dt.date
df_forecast[['business_unit1','business_unit2','process_1']]=df_forecast[['business_unit1','business_unit2','process_1']].apply(lambda x:x.str.upper())
df_rate[['business_unit2','process_1','process_2']]=df_rate[['business_unit2','process_1','process_2']].apply(lambda x:x.str.upper())

df_forecast.head()
df_rate.head()

st.subheader("Data view")
col1,col2=st.columns(2)

with col1:
    col1.subheader("Forecast preview")
    st.dataframe(df_forecast)

with col2:
    col2.subheader("Process rates")
    st.dataframe(df_rate)


# start the model run

button_result=st.button("Run the model",type="primary")
if button_result==False:
    st.stop()
if button_result==True:

    def calculate_hours(df1,df2):
        df = df1.merge(df2,on=['business_unit1','business_unit2','process_1'],how='left')
        return df

    df_plan=calculate_hours(df_forecast,df_rate)


    df_plan['labor_hours']=df_plan['forecast'] / df_plan['unit_rate']
    df_plan['headcount']=df_plan['labor_hours'] / float(shift_hrs)
    df_plan['headcount']=np.ceil(df_plan['headcount'])
    df_inbound=df_plan[df_plan['process_1'].isin(['inbound'])]
    df_outbound=df_plan[df_plan['process_1'].isin(['outbound'])]

    start_date=df_plan['date'].min()
    start_date1=start_date

    df=df_plan[df_plan['date']>=start_date1]
    df['week']=df['date'].apply(lambda x:pd.to_datetime(x)).dt.strftime('%Y-%U')
    df1=df.groupby(by=['week','process_1'],as_index=False).agg({'labor_hours':'sum',
                                                                    'headcount':'mean'})
    df1[['headcount','labor_hours']]=df1[['headcount','labor_hours']].apply(lambda x:np.ceil(x))

    st.subheader("Weekly hours by process")
    st.bar_chart(df1,x='week',y='labor_hours',color='process_1')

    st.subheader("Weekly headcount by process")
    st.bar_chart(df1,x='week',y='headcount',color='process_1')

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv=convert_df(df1)

st.download_button(label="Download model results",data=csv,file_name='labor plan output.csv',mime='text/csv')

st.write("Thank you for visiting our model today! Have a nice day")
