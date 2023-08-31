import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def inflation():
    # Step 1: Read the inflation data from CSV
    df_inflation = pd.read_csv('inflation.csv')

    # Step 2: Data Preprocessing
    df_inflation.rename(columns={'Year': 'ds', 'Inflation Rate (%)': 'y'}, inplace=True)
    df_inflation['ds'] = pd.to_datetime(df_inflation['ds'], format='%Y')  # Convert to datetime format

    # Filter data for the range 2000-2023
    start_date = pd.to_datetime('1962-01-01')
    end_date = pd.to_datetime('2023-12-31')
    df_filtered = df_inflation[(df_inflation['ds'] >= start_date) & (df_inflation['ds'] <= end_date)]

    # Step 3: Prophet Forecasting
    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df_filtered)

    # Create future dataframe for forecasting 2024-2043
    forecast_years = pd.date_range(start='1962-01-01', end='2033-12-31', freq='Y')
    future = pd.DataFrame({'ds': forecast_years})

    forecast = model.predict(future)

    plot1 = m.plot(forecast)

    # Step 4: Plotting Figures
    # Plot historical data from 2000 to 2023 and forecasted values from 2024 to 2043
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_filtered['ds'], df_filtered['y'], label='Actual Inflation Rate', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Inflation Rate', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlim(pd.to_datetime('2000-01-01'), pd.to_datetime('2033-12-31'))
    ax.set_xlabel('Year')
    ax.set_ylabel('Inflation Rate (%)')
    ax.set_title('Actual and Forecasted Inflation Rate')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.show()

    return(fig1,fig2)
