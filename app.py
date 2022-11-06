from stocks import *
from functions import *
from datetime import datetime
import streamlit as st

st.set_page_config(layout="wide")

st.title("Tech Stocks Trading Assistant")

left_column, right_column = st.columns(2)

with left_column:

    all_tickers = {
                "Apple":"AAPL", 
                "Microsoft":"MSFT", 
                "Nvidia":"NVDA", 
                "Paypal":"PYPL",
                "Amazon":"AMZN",
                "Spotify":"SPOT",
                #"Twitter":"TWTR",
                "AirBnB":"ABNB",
                "Uber":"UBER",
                "Google":"GOOG"
                }

    st.subheader("Technical Analysis Methods")
    option_name = st.selectbox('Choose a stock:', all_tickers.keys())
    option_ticker = all_tickers[option_name]
    execution_timestamp = datetime.now()
    'You selected: ', option_name, "(",option_ticker,")"
    'Last execution:', execution_timestamp

    s = Stock_Data()
    t = s.Ticker(tick=option_ticker)  

    m = Models()

    with st.spinner('Loading stock data...'):

        technical_analysis_methods_outputs = {
            'Technical Analysis Method': [
                'Bollinger Bands (20 days & 2 stand. deviations)', 
                'Bollinger Bands (10 days & 1.5 stand. deviations)', 
                'Bollinger Bands (50 days & 3 stand. deviations)', 
                'Moving Average Convergence Divergence (MACD)'
                ],
                'Outlook': [
                    m.bollinger_bands_20d_2std(t),         
                    m.bollinger_bands_10d_1point5std(t), 
                    m.bollinger_bands_50d_3std(t), 
                    m.MACD(t)
                    ],
                'Timeframe of Method': [
                    "Medium-term",         
                    "Short-term", 
                    "Long-term", 
                    "Short-term"
                    ]
                }

        df = pd.DataFrame(technical_analysis_methods_outputs)


    def color_survived(val):
        color = ""
        if (val=="Sell" or val=="Downtrend and sell signal" or val=="Downtrend and no signal"):
            color="#EE3B3B"
        elif (val=="Buy" or val=="Uptrend and buy signal" or val=="Uptrend and no signal"):
            color="#3D9140"
        else:
            color="#CD950C"
        return f'background-color: {color}'
    
    
    st.table(df.sort_values(['Timeframe of Method'], ascending=False).
               reset_index(drop=True).style.applymap(color_survived, subset=['Outlook']))
    
with right_column:

    st.subheader("FinBERT-based Sentiment Analysis")

    with st.spinner("Connecting with www.marketwatch.com..."):
        st.plotly_chart(m.finbert_headlines_sentiment(t)["fig"])
        "Current sentiment:", m.finbert_headlines_sentiment(t)["current_sentiment"], "%"

    st.subheader("LSTM-based 7-day stock price prediction model")

    with st.spinner("Compiling LSTM model.."):
        st.plotly_chart(m.LSTM_7_days_price_predictor(t))
        