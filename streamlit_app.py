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

#Reading the forecast template from the source to make it available as sample to download for the user
forecast_template = pd.read_csv("input_files/forecast_input_file.csv")
#forecast_template = pd.read_csv("forecast_input_file.csv")
forecast_template_str = forecast_template.to_csv(index=False)  
st.sidebar.header("Forecasts")
st.sidebar.download_button(label="Click to download a forecast template",data=forecast_template_str,file_name='forecast_template.csv',mime='text/csv')

#Providing option to upload the user's forecast data in the form of forecast template provided above
st.sidebar.subheader("Please upload your data here")
forecast_file=st.sidebar.file_uploader("Upload forecast file")


# Initialize session state for the number of rows
if "num_rows" not in st.session_state:
    st.session_state.num_rows = 3
if "df_forecast" not in st.session_state:
        st.session_state.df_forecast = False
if "df_rate" not in st.session_state:
        st.session_state.df_rate = False
if "shift_hrs" not in st.session_state:
        st.session_state.shift_hrs = False

# data input
with st.form(key='Rate Form'):
			col1,col2,col3 = st.columns([1,2,3])

			with col1:
				amount = st.number_input("Shift hours per day",1,24)

			with col2:
				hour_per_week = st.number_input("Labor rate per hour in USD",1,120)
                        
			with col3:
				calculated = st.form_submit_button(label='calculate')
                    
if calculated:
    #shift_hrs=st.text_input('Shift hours per day','8')
    st.session_state.shift_hrs = hour_per_week

def calculate_hours(df1,df2):
    df = df1.merge(df2,on=['Business','Area','Process'],how='left')
    return df

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# Function to add a new row
def add_row():
    st.session_state.num_rows += 1

# Function to remove the last row
def remove_row():
    if st.session_state.num_rows > 1:
        st.session_state.num_rows -= 1

# Column names
columns = ["S.No.", "Business", "Area", "Process","Function","Unit_Rate","Percentage Allocation"]

# Create a dictionary to hold the form data, including serial numbers
data = {col: [""] * st.session_state.num_rows for col in columns}
data["S.No."] = list(range(1, st.session_state.num_rows + 1))  # Serial numbers

# Display the form in a tabular format
with st.form("data_entry_form"):
    st.write("Enter your Process rate data:")

    # Create a table header
    cols = st.columns(len(columns))
    for col, column_name in zip(cols, columns):
        col.write(f"**{column_name}**")  # Make the header bold

    # Create inputs for each cell in the table, excluding the serial number column
    for i in range(st.session_state.num_rows):
        cols = st.columns(len(columns))
        for j, (col, column_name) in enumerate(zip(cols, columns)):
            if column_name == "S.No.":
                col.write(data[column_name][i])
            else:
                data[column_name][i] = col.text_input(f"{column_name}_{i}", value=data[column_name][i], label_visibility="collapsed")

    # Buttons to add or remove rows within the form
    col1, col2, col3 = st.columns([1, 2, 3])
    with col1:
        st.form_submit_button("Add Row", on_click=add_row)
    with col2:
        submitted = st.form_submit_button("Submit")
    with col3:
        st.form_submit_button("Remove Row", on_click=remove_row)

# Process the form data after submission
if submitted:
    if forecast_file is None:
        st.info("Please Upload a forecast file through config")
        st.stop()

    else:
        # Convert the dictionary to a DataFrame
        df_rate = pd.DataFrame(data)
        df_rate = df_rate.drop(columns = ['S.No.'])
        df_forecast=pd.read_csv(forecast_file,skiprows=1,names=['business_unit1','business_unit2','date','process_1','forecast'])

        #Formatting forecast data
        df_forecast['date']=pd.to_datetime(df_forecast['date']).dt.date
        df_forecast[['business_unit1','business_unit2','process_1']]=df_forecast[['business_unit1','business_unit2','process_1']].apply(lambda x:x.str.upper())
        df_forecast = df_forecast.rename(columns = {'business_unit1':'Business','business_unit2':'Area','process_1':'Process'})
        st.session_state.df_forecast = df_forecast
        df_rate[["Business", "Area", "Process","Function","Unit_Rate","Percentage Allocation"]]=df_rate[["Business", "Area", "Process","Function","Unit_Rate","Percentage Allocation"]].apply(lambda x:x.str.upper())
        df_rate['Unit_Rate'] = df_rate['Unit_Rate'].astype('int')
        st.session_state.df_rate = df_rate
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
    df_plan=calculate_hours(st.session_state.df_forecast,st.session_state.df_rate)
    df_plan['labor_hours']=df_plan['forecast'] / df_plan['Unit_Rate']
    df_plan['headcount']=df_plan['labor_hours'] / float(st.session_state.shift_hrs)
    df_plan['headcount']=np.ceil(df_plan['headcount'])
    df_inbound=df_plan[df_plan['Process'].isin(['inbound'])]
    df_outbound=df_plan[df_plan['Process'].isin(['outbound'])]

    start_date=df_plan['date'].min()
    start_date1=start_date

    df=df_plan[df_plan['date']>=start_date1]
    df['week']=df['date'].apply(lambda x:pd.to_datetime(x)).dt.strftime('%Y-%U')
    df1=df.groupby(by=['week','Process'],as_index=False).agg({'labor_hours':'sum',
                                                                    'headcount':'mean'})
    df1[['headcount','labor_hours']]=df1[['headcount','labor_hours']].apply(lambda x:np.ceil(x))

    st.subheader("Weekly hours by process")
    st.bar_chart(df1,x='week',y='labor_hours',color='Process')

    st.subheader("Weekly headcount by process")
    st.bar_chart(df1,x='week',y='headcount',color='Process')

    csv=convert_df(df1)
    st.download_button(label="Download model results",data=csv,file_name='labor plan output.csv',mime='text/csv')
    st.write("Thank you for visiting our model today! Have a nice day")