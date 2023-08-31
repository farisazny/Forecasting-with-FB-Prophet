import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def labour():

    # Step 1: Read the dataset from CSV
    df = pd.read_csv('Users/User/Documents/GitHub/FYP1 GitHub/Streamlit/data_labor.csv')  

    # Step 2: Data Preprocessing
    category = 'Unemployment Rate'  # Specify the category column

    df = df[['Year', category]].copy()
    df.rename(columns={'Year': 'ds', category: 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')

    # Step 3: Prophet Forecasting
    m = Prophet(
        growth='linear',
        yearly_seasonality=True
    )
    model = m.fit(df)

    future = model.make_future_dataframe(periods=12, freq='Y')  # Forecasting for 1 year (12 months)
    forecast = model.predict(future)

    # Plotting Figures
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ds'], df['y'], label='Actual', marker='o')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted', color='orange')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_xlabel('Year')
    ax.set_ylabel(category)
    ax.set_title(f'Actual and Forecasted {category}')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    plt.show()

    return(fig1,fig2)
