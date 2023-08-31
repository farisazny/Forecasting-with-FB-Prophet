import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def cpi_state(state):
    # Step 1: Read the CPI data from CSV
    df_cpi = pd.read_csv('FBProphet/cpi_state.csv')

    # Step 2: Data Preprocessing
    # Assuming your data is already in the required format, no column renaming or datetime conversion is needed.

    # Step 3: Prophet Forecasting for 'Housing / Utilities' Category for Johor
    category = 'Housing / Utilities'  # Specify the category of interest
    

    


    df_state = df_cpi[(df_cpi['Category'] == category) & (df_cpi[state].notna())][['Date', state]].copy()
    df_state.rename(columns={'Date': 'ds', state: 'y'}, inplace=True)
    df_state['ds'] = pd.to_datetime(df_state['ds'], format='%m/%d/%Y')

    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df_state)

    future = model.make_future_dataframe(periods=12, freq='M')  # Forecasting for 1 year (12 months)
    forecast = model.predict(future)

    # Plotting Figures for Johor
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_state['ds'], df_state['y'], label=f'Actual {category} CPI in {state}', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label=f'Forecasted {category} CPI in {state}', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI')
    ax.set_title(f'Actual and Forecasted {category} CPI in {state}')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot forecast components for Johor
    fig2 = model.plot_components(forecast)
    plt.show()

    return(fig1, fig2)