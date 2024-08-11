from datetime import datetime, timedelta, date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
plt.style.use('dark_background')


def risk_return_plot(df: pd.DataFrame, x: str, y: str, historical_years: int, time_freq: str):
    # Create scatter plot
    sns.scatterplot(
        data=df, x=x, y=y, 
        color='lightblue', legend=None, s=65, edgecolor='white',
    )

    # Annotate the symbols
    symbol_positions = {}
    for index, row in df.iterrows():
        x_val, y_val = row[x], row[y]
        symbol = row['Symbol']
        
        if (x_val, y_val) in symbol_positions.values():
            for i in range(1, len(symbol) + 1):
                if (x_val, y_val - i * 0.05) not in symbol_positions.values():
                    y_val -= i * 0.05
                    break
        symbol_positions[symbol] = (x_val, y_val)
        plt.text(x_val, y_val, symbol, ha='right', va='top', fontsize=8, color='white')

    # Add axis labels and title
    plt.xlabel('Total Risk', fontsize=9)
    plt.ylabel('Average Return (%)', fontsize=9)
    plt.title('Historical Risk & Return', fontsize=11)

    # Add text box with time frame and period
    textstr = f'Time frame: {historical_years} yr(s)\nInterval: {time_freq}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gcf().text(0.05, -0.025, textstr, fontsize=9, bbox=props)
    
    # Show plot
    plt.show()
