from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


class StockAnalysis:
    def __init__(self, symbol: str, start_date: datetime, end_date: datetime, freq: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.stock_df = self._get_stock_data_w_returns()
    
    def _get_stock_data_w_returns(self) -> pd.DataFrame:
        # Retrieves stock data for the given ticker and calculates returns
        stock_df = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.freq, rounding=True)
        stock_df['Returns(%)'] = (stock_df['Adj Close'].pct_change() * 100).round(2)
        stock_df.reset_index(inplace=True) 
        stock_df.dropna(inplace=True)
        return stock_df

    def calculate_average_return(self) -> float:
        # Calculates the average returns for the specified stock symbol
        return self.stock_df['Returns(%)'].mean()

    def get_stock_beta(self) -> float:
        # Calculates stock beta against SPY
        spy_analysis = StockAnalysis('SPY', self.start_date, self.end_date, self.freq)
        spy_returns = spy_analysis.stock_df['Returns(%)']
        cov_matrix = np.cov(self.stock_df['Returns(%)'], spy_returns)
        cov_stock_spy = cov_matrix[0, 1]
        var_spy = np.var(spy_returns)
        beta = cov_stock_spy / var_spy
        return beta.round(2)

    def get_stock_std(self) -> float:
        # Calculates stock standard deviation
        return np.sqrt(np.var(self.stock_df['Returns(%)']))

    def get_analysis_summary(self) -> pd.DataFrame:
        # Compiles analysis results into a DataFrame
        summary = {
            'Symbol': [self.symbol],
            'Average_Return(%)': [self.calculate_average_return()],
            'Beta': [self.get_stock_beta()],
            'Standard_Deviation(%)': [self.get_stock_std()],
            'Total_Risk': [self.get_stock_std() + self.get_stock_beta()]
        }
        return pd.DataFrame(summary)

def analyze_stocks(symbols: str, start_date: datetime, end_date: datetime, freq: str) -> pd.DataFrame:
    all_summaries = []

    for symbol in symbols:
        analysis = StockAnalysis(symbol, start_date, end_date, freq)
        summary_df = analysis.get_analysis_summary()
        all_summaries.append(summary_df)
    
    combined_df = pd.concat(all_summaries, ignore_index=True)
    return combined_df
# print('run')

class StockRatios:
    def __init__(self, symbols: pd.Series):
        self.symbols = symbols
        self.etf_list = self.get_etfs(symbols)
        self.stocks = self.get_stocks(symbols, self.etf_list)
        self.ratios = pd.DataFrame()  # Initialize an empty DataFrame to store ratios

    def get_etfs(self, symbols):
        etf_list = [] 
        for stock in symbols:
            try:
                ticker = yf.Ticker(stock)
                income_statement = ticker.quarterly_financials.iloc[:, [0]]
            except (IndexError, KeyError):
                etf_list.append(stock)
        return etf_list
    
    def get_stocks(self, symbols, etf_list):
        return [i for i in symbols if i not in etf_list]
    
    def compute_gross_profit_margin(self):
        results = []
        for stock in self.stocks:
            try:
                ticker = yf.Ticker(stock)
                income_statement = ticker.quarterly_financials.iloc[:, [0]]
                date = income_statement.columns[0]
                gross_profit_margin = ((income_statement.loc['Gross Profit'] / income_statement.loc['Total Revenue']) * 100).values[0]
                results.append({
                    'Symbol': ticker.ticker,
                    'Date': date,
                    'Gross_Profit_Margin': gross_profit_margin
                })
            except (IndexError, KeyError):
                continue
        return results

    def get_ratio_df(self):
        # Aggregate all ratios into a single list of results
        all_results = []
        all_results.extend(self.compute_gross_profit_margin())
        
        # Convert list of results to a DataFrame
        ratio_df = pd.DataFrame(all_results)
        
        # Set 'Date' as the index
        if 'Date' in ratio_df.columns:
            ratio_df.set_index('Date', inplace=True)
        
        self.ratios = ratio_df
        return self.ratios