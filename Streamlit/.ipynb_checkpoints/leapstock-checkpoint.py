import yfinance as yf
import streamlit as st
import pandas as pd
import hvplot.pandas
from bokeh.plotting import figure
import holoviews as hv
hv.extension('bokeh', logo=False)
from PIL import Image
import datetime

start = datetime.datetime(2017,1,31) 
end = datetime.datetime(2021,1,31) 

st.set_page_config(layout="wide")

tradingsymbol_md_1 = ""
tradingsymbol_md_2 = ""
tradingsymbol_md_3 = ""
tradingsymbol_md_4 = ""

plot_1 =""
plot_2 =""
plot_3 =""
plot_4 =""

# Load Leap Image
image = Image.open('Leap.png')
st.sidebar.image(image, use_column_width=True)

# load trading symbols list csv
tradingsymbols_list_df = pd.read_csv('Tradingsymbols List.csv')

st.title("LEAPS vs Stocks Trading Predictor")
st.markdown("This application predicts LEAPS against Stock investing for selected trading symbols")

st.sidebar.markdown("Select one or more trading symbols")

tradingsymbol_sel = st.sidebar.multiselect('', tradingsymbols_list_df['Trading Symbols'].unique())

url1_md = f"[What are LEAPS Options?](https://tickertape.tdameritrade.com/trading/what-are-leaps-options-15074)"
url2_md = f"[Options Trading Basics Explained](https://www.youtube.com/watch?v=ipf0Yg2Z4Gs)"
url3_md = f"[Stock Price Data, Financial and Stock Market API](https://eodhistoricaldata.com/financial-apis/stock-options-data)"
url4_md = f"[How to get stock data using python](https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75)"

st.sidebar.title("Resources")
st.sidebar.markdown(url1_md)
st.sidebar.markdown(url2_md)
st.sidebar.markdown(url3_md)
st.sidebar.markdown(url4_md)

column_1, column_2, column_3 = st.beta_columns([1,1,2])

i=0

if tradingsymbol_sel:
    for file in tradingsymbol_sel:

        i+=1
       
        path = file + ".csv"
        jpg = file + ".jpg"
        
        if i==1:
            tradingsymbol_md_1 = f"{file}"
            tradingsymbol_prices_1 = pd.read_csv(path, index_col="Month", infer_datetime_format=True, parse_dates=True)
            tradingsymbol_prices_1 = tradingsymbol_prices_1.sort_index()

            tradingsymbol_model_1 = file
            tradingsymbol_title_1 = f"LSTM Trading Symbol Model {tradingsymbol_model_1}: LEAPS vs Stocks"

            # Plot the LEAPS vs Stocks values as a line chart
            plot_1 = tradingsymbol_prices_1.hvplot.line(xlabel="Month",
                                   ylabel="Price",
                                   title=tradingsymbol_title_1)
            
            #define the ticker symbol
            tickerSymbol = 'TSLA'

            #get data on this ticker
            tickerData = yf.Ticker(tickerSymbol)

            #get the historical prices for this ticker
            tickerDf_1 = tickerData.history(start=start, end=end)
           
        elif i==2:
            tradingsymbol_md_2 = f"{file}"            
            tradingsymbol_prices_2 = pd.read_csv(path, index_col="Month", infer_datetime_format=True, parse_dates=True)
            tradingsymbol_prices_2 = tradingsymbol_prices_2.sort_index()

            tradingsymbol_model_2 = file
            tradingsymbol_title_2 = f"LSTM Trading Symbol Model {tradingsymbol_model_2}: LEAPS vs Stocks"

            # Plot the LEAPS vs Stocks values as a line chart
            plot_2 = tradingsymbol_prices_2.hvplot.line(xlabel="Month",
                                   ylabel="Price",
                                   title=tradingsymbol_title_2)
            
            #define the ticker symbol
            tickerSymbol = 'NVDA'

            #get data on this ticker
            tickerData = yf.Ticker(tickerSymbol)

            #get the historical prices for this ticker
            tickerDf_2 = tickerData.history(start=start, end=end)
            
        elif i==3:
            tradingsymbol_md_3 = f"{file}"            
            tradingsymbol_prices_3 = pd.read_csv(path, index_col="Month", infer_datetime_format=True, parse_dates=True)
            tradingsymbol_prices_3 = tradingsymbol_prices_3.sort_index()

            tradingsymbol_model_3 = file
            tradingsymbol_title_3 = f"LSTM Trading Symbol Model {tradingsymbol_model_3}: LEAPS vs Stocks"

            # Plot the LEAPS vs Stocks values as a line chart
            plot_3 = tradingsymbol_prices_3.hvplot.line(xlabel="Month",
                                   ylabel="Price",
                                   title=tradingsymbol_title_3)

            #define the ticker symbol
            tickerSymbol = 'SHOP'

            #get data on this ticker
            tickerData = yf.Ticker(tickerSymbol)

            #get the historical prices for this ticker
            tickerDf_3 = tickerData.history(start=start, end=end)

        elif i==4:
            tradingsymbol_md_4 = f"{file}"            
            tradingsymbol_prices_4 = pd.read_csv(path, index_col="Month", infer_datetime_format=True, parse_dates=True)
            tradingsymbol_prices_4 = tradingsymbol_prices_4.sort_index()

            tradingsymbol_model_4 = file
            tradingsymbol_title_4 = f"LSTM Trading Symbol Model {tradingsymbol_model_4}: LEAPS vs Stocks"

            # Plot the LEAPS vs Stocks values as a line chart
            plot_4 = tradingsymbol_prices_4.hvplot.line(xlabel="Month",
                                   ylabel="Price",
                                   title=tradingsymbol_title_2)

            #define the ticker symbol
            tickerSymbol = 'SNPS'

            #get data on this ticker
            tickerData = yf.Ticker(tickerSymbol)

            #get the historical prices for this ticker
            tickerDf_4 = tickerData.history(start=start, end=end)
            
        else:
            st.write("Max 4 trading symbols allowed")

    if plot_1:
        
        with column_1:
            st.header("Close")
            st.line_chart(tickerDf_1.Close)
        with column_2:
            st.header("Volume")
            st.line_chart(tickerDf_1.Volume)
        with column_3:
            st.header(tradingsymbol_md_1)
            st.write(hv.render(plot_1, backend='bokeh'))
    if plot_2:
        
        with column_1:
            st.header("Close")
            st.line_chart(tickerDf_2.Close)
        with column_2:
            st.header("Volume")
            st.line_chart(tickerDf_2.Volume)   
        with column_3:            
            st.header(tradingsymbol_md_2)
            st.write(hv.render(plot_2, backend='bokeh'))
    if plot_3:
        
        with column_1:
            st.header("Close")
            st.line_chart(tickerDf_3.Close)
        with column_2:
            st.header("Volume")
            st.line_chart(tickerDf_3.Volume)   
        with column_3:
            st.header(tradingsymbol_md_3)
            st.write(hv.render(plot_3, backend='bokeh'))
    if plot_4:
        
        with column_1:
            st.header("Close")
            st.line_chart(tickerDf_4.Close)
        with column_2:
            st.header("Volume")
            st.line_chart(tickerDf_4.Volume)   
        with column_3:          
            st.header(tradingsymbol_md_4)
            st.write(hv.render(plot_4, backend='bokeh'))

else:
    st.write("No trading symbol selected")


    
#if tradingsymbol_sel:
#     with column_1:
#         st.header(tradingsymbol_md_1)
#         #st.write(plot_1) #, backend='bokeh', use_column_width=True))
#         st.write(hv.render(plot_1, backend='bokeh'))
#     with column_2:
#         st.header(tradingsymbol_md_2)
#         #st.write(plot_2) #, backend='bokeh', use_column_width=True))
#         st.write(hv.render(plot_2, backend='bokeh'))        
#     with column_3:
#         st.header(tradingsymbol_md_3)
#         #st.write(plot_3) #, backend='bokeh', use_column_width=True)
#         st.write(hv.render(plot_3, backend='bokeh'))        
#     with column_4:
#         st.header(tradingsymbol_md_4)
#         #st.write(plot_4) #, backend='bokeh', use_column_width=True)
#         st.write(hv.render(plot_4, backend='bokeh'))        
  
#https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75

#define the ticker symbol
#tickerSymbol = 'GOOGL'

#get data on this ticker
#tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
#tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

#st.line_chart(tickerDf.Close)
#st.line_chart(tickerDf.Volume)

