from configparser import ParsingError
from logging import raiseExceptions
import yfinance as yf
import requests
import pandas as pd
from bs4 import BeautifulSoup

class Stock_Data(object):        
        '''
        This class contains 5 methods responsible for choosing a stock's ticker, then checking whether the 
        stock exchange it is listed in is open or not, and in case it is, it gets data for the last 6 months 
        from "yfinance" module of Yahoo Inc. which will be fed to the models.
        '''
        
        def Ticker(self, tick):
                '''
                This method will "carry" the company's ticker, and it will also be used as a placeholder.
                '''
                global ticker 
                ticker = tick

                return ticker


        def status_getter(self, Ticker):
                '''
                This method gets the company ticker the user chooses, creates a www.marketwatch.com
                link, then scraps the HTML code of the corresponding company page in marketwatch website,
                and gets the current market status of the exchange this stock is listed in. Possible values are:
                After Hours, Open, and Market Closed. 
                '''
                global company_ticker
                company_ticker = Ticker 
                link_1 = 'https://www.marketwatch.com/investing/stock/'
                link_2 = '?mod=search_symbol'
                # Pasting the above 3 parts to create the URL
                global final_link
                final_link = link_1 + company_ticker + link_2
                
                page = requests.get(final_link)
                global soup
                soup = BeautifulSoup(page.text, "lxml")
                if soup is None:
                        raise ParsingError("HTML code of MarketWatch website was not scraped and current status can not be found")
                else:
                        current_status = soup.find("div", class_="status").text # Finding the market status
                        return current_status


        def current_price_getter(self, Ticker):
                '''
                This method will get the current price only if the market is open.
                '''
                current_price = None 
                if self.status_getter(Ticker) == "Open":
                        current_price = float(soup.find("bg-quote", class_="value").text.replace(',',''))
                        return current_price
                else:
                        return "Market Closed"

        def stock_data_getter(self, Ticker):
                '''
                This method will return a dataframe containing Stock data from the Yahoo's "yfinance" 
                library in case the market is open.
                '''
                if self.status_getter(Ticker) == "Open":
                        data = yf.download(tickers = str(Ticker), period = "6mo", interval = "1d")
                        df = pd.DataFrame(data)
                        return df
                else:
                        return "Market Closed"

        def LSTM_stock_data_getter(self, Ticker):
                '''
                This method will return a dataframe containing Stock data from the Yahoo's "yfinance" 
                library regardrless of whether the market is open or not, and will feed the LSTM model.
                '''
                data = yf.download(tickers = str(Ticker), period = "2y", interval = "1d")
                df = pd.DataFrame(data)
                            
                # Prediction in the data we evaluate the model
                # If the user wants to run the model with the data that has been evaluated and predicted for , uncomment the 2 lines below ( Rows 111-112 )
                # Setting the start = 2022-08-26 and end = 2020-08-26 Yahoo Finance will return data from 25-8-2020 to 25-8-2022 (2 years period).
                # In those data our model has been evaluated.

                #data = yf.download(tickers = str(Ticker),end="2022-08-26", start="2020-08-26") 
                #df = pd.DataFrame(data)
                
                return df
                

        def article_parser(self, ticker):
                '''
                This method gets as input a stock ticker, creates the www.marketwatch.com link of this stock
                and returns a dataframe with the last 17 articles' headers.
                '''
                company_ticker = self.Ticker(tick=ticker)  
                link_1 = 'https://www.marketwatch.com/investing/stock/'
                link_2 = '?mod=search_symbol'
                # Pasting the above 3 parts to create the URL
                final_link = link_1 + company_ticker + link_2


                page = requests.get(final_link)
                soup = BeautifulSoup(page.content, "html.parser")
                results = soup.find("div", class_="tab__pane is-active j-tabPane")
                articles = results.find_all("a", class_="link")

                headerList = ["ticker", "headline"]
                rows = []
                counter = 1
                df_headers = pd.DataFrame()

                for art in articles:
                        if counter <= 17:
                                ticker = company_ticker
                                title = art.text.strip()
                                if title is None:
                                        break
                                rows.append([ticker, title])
                                counter = counter + 1

                        df_headers = pd.DataFrame(rows, columns=headerList)

                return df_headers
                
