import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def exchange(Category):
    # Step 1: Read the exchange rate data from CSV
    df_exchange = pd.read_csv('data_exchangerate.csv')

    # Step 2: Data Preprocessing
    
    df_forecast = df_exchange[['Date', Category]].copy()
    df_forecast.rename(columns={'Date': 'ds', Category: 'y'}, inplace=True)
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], format='%d/%m/%Y')

    # Step 3: Prophet Forecasting
    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df_forecast)

    future = model.make_future_dataframe(periods=365, freq='D')  # Forecasting for 1 year
    forecast = model.predict(future)

    # Plotting Figures
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_forecast['ds'], df_forecast['y'], label='Actual Exchange Rate', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Exchange Rate', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.set_title(f'Actual and Forecasted Exchange Rate ({Category})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.show()

    return(fig1, fig2)