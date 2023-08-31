import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def gdp(Series_Type):
    # Step 1: Read the GDP data from CSV
    df_gdp = pd.read_csv('FBProphet/gdp.csv')

    # Step 2: Data Preprocessing
    
    Category = 'Overall (RM mil)'
    df_real_overall = df_gdp[df_gdp['Series Type'] == Series_Type][['Date', Category]].copy()
    df_real_overall.rename(columns={'Date': 'ds', Category: 'y'}, inplace=True)
    df_real_overall['ds'] = pd.to_datetime(df_real_overall['ds'], format='%m/%d/%Y')

    # Set the frequency to 'Q' for quarterly data
    df_real_overall.set_index('ds', inplace=True)
    df_real_overall = df_real_overall.resample('Q').sum().reset_index()

    # Step 3: Prophet Forecasting for  Category
    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df_real_overall)

    future = model.make_future_dataframe(periods=12, freq='Q')  # Forecasting for 1 year (12 quarters)
    forecast = model.predict(future)


    # Plotting Figures
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_real_overall['ds'], df_real_overall['y'], label='Actual GDP', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted GDP', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('GDP (Overall RM mil)')
    ax.set_title('Actual and Forecasted GDP (Overall RM mil)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.show()
    return(fig1, fig2)