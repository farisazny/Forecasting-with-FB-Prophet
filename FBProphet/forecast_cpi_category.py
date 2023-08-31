import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def cpi_category(Category):
    # Step 1: Read the CPI data from CSV
    df_cpi = pd.read_csv('cpi_category.csv')

    # Step 2: Data Preprocessing
    
    df_cpi.rename(columns={'Date': 'ds', Category: 'y'}, inplace=True)
    df_cpi['ds'] = pd.to_datetime(df_cpi['ds'], format='%m/%d/%Y')  # Convert to datetime format

    # Step 3: Prophet Forecasting
    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df_cpi)

    # Create future dataframe for forecasting
    forecast_periods = 240  # Forecasting for 20 years (240 months)
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')

    forecast = model.predict(future)

    # Step 4: Plotting Figures
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_cpi['ds'], df_cpi['y'], label='Actual Housing CPI', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Housing CPI', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Housing CPI')
    ax.set_title('Actual and Forecasted Housing Consumer Price Index (CPI)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.show()

    return(fig1, fig2)
