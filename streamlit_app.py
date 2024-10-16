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
#st.write("This is a prototype tool in development to compute labor planning requirement based on input parameters")

#Reading the forecast template from the source to make it available as sample to download for the user
forecast_template = pd.read_csv("input_files/forecast_input_file.csv")
#forecast_template = pd.read_csv("forecast_input_file.csv")
forecast_template_str = forecast_template.to_csv(index=False)  
st.sidebar.header("Forecast")
st.sidebar.download_button(label="Click to download template",data=forecast_template_str,file_name='forecast_template.csv',mime='text/csv')

#Providing option to upload the user's forecast data in the form of forecast template provided above
st.sidebar.subheader("Upload Forecast data")
forecast_file=st.sidebar.file_uploader("")#Upload forecast file


# Initialize session state for the number of rows
if "num_rows" not in st.session_state:
    st.session_state.num_rows = 3
if "df_forecast" not in st.session_state:
    st.session_state.df_forecast = False
if "df_rate" not in st.session_state:
    st.session_state.df_rate = False
if "shift_hrs" not in st.session_state:
    st.session_state.shift_hrs = False
if "labor_rate" not in st.session_state:
    st.session_state.labor_rate = False
if "start_date" not in st.session_state:
    st.session_state.start_date = None
if "end_date" not in st.session_state:
    st.session_state.end_date = None
if "submit_state" not in st.session_state:
    st.session_state.submit_state = False

# data input
with st.form(key='Rate Form'):
			col1,col2,col3 = st.columns([1,2,3])

			with col1:
				hrs = st.number_input("Shift hours per day",1,24)

			with col2:
				l_cost = st.number_input("Labor rate per hour in USD",1,120)
                        
			with col3:
				calculated = st.form_submit_button(label='calculate')
        
if calculated:
    #shift_hrs=st.text_input('Shift hours per day','8')
    st.session_state.shift_hrs = hrs
    st.session_state.labor_rate=l_cost

def calculate_hours(df1,df2):
    df = df1.merge(df2,on=['Business','Area','Process'],how='left')
    return df

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# Function to add a new row
def add_row():
    st.session_state.submit_state = False
    st.session_state.num_rows += 1

# Function to remove the last row
def remove_row():
    if st.session_state.num_rows > 1:
        st.session_state.num_rows -= 1
        
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

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
if submitted or st.session_state.submit_state:
    st.session_state.submit_state = True
    if forecast_file is None:
        st.info("Please Upload a forecast file through config")
        st.stop()

    else:
        df_forecast=pd.read_csv(forecast_file,skiprows=1,names=['business_unit1','business_unit2','date','Process','forecast'])
        
        #Formatting forecast data and storing in session state
        df_forecast['date']=pd.to_datetime(df_forecast['date']).dt.date
        df_forecast[['business_unit1','business_unit2','Process']]=df_forecast[['business_unit1','business_unit2','Process']].apply(lambda x:x.str.upper())
        df_forecast = df_forecast.rename(columns = {'business_unit1':'Business','business_unit2':'Area','Process':'Process'})
        

        min_date=df_forecast['date'].min()
        max_date=df_forecast['date'].max()

        #Providing date range filter
        st.title("Select date range")
        col1,col2=st.columns(2)
        
        with col1:
            st.session_state.start_date=st.date_input("Pick start date",value=min_date,min_value=min_date,max_value=max_date)

        with col2:
            st.session_state.end_date=st.date_input("Pick end date",value=max_date,min_value=min_date,max_value=max_date)
        
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        
        if end_date<=start_date:
            st.warning('Please select a end date that is after start date',icon="⚠️")
            st.stop()
        
        else:
            #Filtered forecast dataframe
            df_forecast=df_forecast[(df_forecast['date']>=start_date) & (df_forecast['date']<=end_date)]
            st.session_state.df_forecast = df_forecast
            # Convert the dictionary to a DataFrame
            df_rate = pd.DataFrame(data)
            df_rate = df_rate.drop(columns = ['S.No.'])

            df_rate[["Business", "Area", "Process","Function","Unit_Rate","Percentage Allocation"]]=df_rate[["Business", "Area", "Process","Function","Unit_Rate","Percentage Allocation"]].apply(lambda x:x.str.upper())
            df_rate['Process']=df_rate['Process'].str.strip()
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
    df_plan['labor_hours']=np.ceil(df_plan['forecast'] / df_plan['Unit_Rate'])
    df_plan['labor_cost']=df_plan['labor_hours']*float(st.session_state.labor_rate)
    df_plan['headcount']=df_plan['labor_hours'] / float(st.session_state.shift_hrs)
    df_plan['headcount']=np.ceil(df_plan['headcount'])
    df_inbound=df_plan[df_plan['Process'].isin(['INBOUND'])]
    df_outbound=df_plan[df_plan['Process'].isin(['OUTBOUND'])]

    start_date=df_plan['date'].min()
    start_date1=start_date

    df=df_plan[df_plan['date']>=start_date1]
    df['week']=df['date'].apply(lambda x:pd.to_datetime(x)).dt.strftime('%Y-%U')
    
    st.write('Model run is complete! Following is the output summary')

    df1=df.groupby(by=['week','Function'],as_index=False).agg({'labor_cost':'sum','labor_hours':'sum','forecast':'sum'})
    df1['weekly_headcount']=df1['labor_hours']/40
    df1[['weekly_headcount','labor_hours','labor_cost']]=df1[['weekly_headcount','labor_hours','labor_cost']].apply(lambda x:np.ceil(x).fillna(0)).astype(int)
    csv=convert_df(df1)

    total_cost=df1['labor_cost'].sum()
    total_units=df1['forecast'].sum()
    #total_cpu=np.round(total_cost/total_units,2)
    total_cpu=total_cost/total_units

    #df2=pd.pivot_table(df1,values=['weekly_headcount','labor_cost'],index=['Function'],columns=['week'],aggfunc=np.sum)
    df2=pd.pivot_table(df1,values='weekly_headcount',index=['Function'],columns=['week'],aggfunc=np.sum)
    df2.loc['TOTAL']=df2.sum(axis=0)

    df3=pd.pivot_table(df,values='labor_hours',index=['Function'],columns=['date'],aggfunc=np.sum)
    df3.loc['TOTAL']=df3.sum(axis=0)

    st.write('Weekly headcount by Function')
    st.table(data=df2)

    st.write('Daily hours by Function')
    st.table(data=df3)

    #st.write('Total cost per unit is USD',total_cpu)
    st.dataframe(df1)
    st.write("total_cost: ",total_cost)
    st.write("total_units: ",total_units)
    st.write(f"Total cost per unit is: ${total_cpu:.4f}")
    
    st.download_button(label="Download model results",data=csv,file_name='labor plan output.csv',mime='text/csv')
    st.write("Thank you for visiting our app today! Have a nice day")
    