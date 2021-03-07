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

# load trading symbols by sector csv
sectorsymbols_df = pd.read_csv('SectorTradingSymbols.csv')

# load strategies csv into df
strategies_df = pd.read_csv('Strategies.csv')

st.title("Sectorwise Investing Strategies Predictor")
st.markdown("This application predicts which strategies perform better for a trading symbol by sector")

st.sidebar.markdown("Select strategies for a symbol within one sector")

sector_df = sectorsymbols_df['Sectors'].unique()
sector_sel = st.sidebar.selectbox('Select Sector:', sector_df)

symbol_df = sectorsymbols_df['SymbolName'].loc[sectorsymbols_df['Sectors'] == sector_sel].unique()
symbol_sel = st.sidebar.selectbox('Select Symbol', symbol_df)

strategies_sel = st.sidebar.multiselect('Select Strategies', strategies_df)

if strategies_sel:
    #path = 'NVDA Nvidia.csv'
    
    #sector_sel = 'Finance'
    #symbol_sel = 'AMT:American Tower Corp'
    selected_symbol = symbol_sel.split(':')

    symbol = selected_symbol[0]

    path = sector_sel + '_' + symbol + '.csv'
       
    tradingsymbol_title_1 = f"Predictor for {symbol_sel} {path}"

    tradingsymbol_prices_1 = pd.read_csv(path, index_col="Month", infer_datetime_format=True, parse_dates=True)
    tradingsymbol_prices_1 = tradingsymbol_prices_1.sort_index()

    strategy_df = tradingsymbol_prices_1.copy()
    
    strategy_df.drop(columns=[col for col in strategy_df if col not in strategies_sel], inplace=True)
    
    # Plot the strategies as a line chart
    plot_1 = strategy_df.hvplot.line(xlabel="Month",
                           ylabel="Price",
                           title=tradingsymbol_title_1,
                           frame_width = 1000,
                           frame_height = 600)

    st.header(tradingsymbol_md_1)
    st.write(hv.render(plot_1, backend='bokeh', ))
    
else:
    st.write("No strategies selected")


