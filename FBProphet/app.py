import streamlit as st
import forecast_cpi_state, forecast_cpi_category, forecast_exchangerate, forecast_gdp, forecast_labor, forecast_inflation  

def main():
    st.title("Forecasting and Visualization")

    task_selection = st.selectbox("Select a Forecasting Task", ["Select One", "Consumer Price Index", "Exchange Rates", "Malaysia GDP", "Labour Stats", "Inflation Rate"])

    if task_selection == "Consumer Price Index":
        
        option = st.selectbox(
        "CPI By State or Category?",
        ('Select One', 'State', 'Category'))
        
        if option == 'State':
            
            optionA = st.selectbox("Select State:", ["Selangor", "Johor", "Terengganu"])
            fig_state, fig_comp_state = forecast_cpi_state.cpi_state(optionA)
            col1, col2 = st.columns(2)
            col1.pyplot(fig_state)
            col2.pyplot(fig_comp_state)

        if option == 'Category':

            optionA = st.selectbox("Select Category:", ["Housing / Utilities", "Clothing / Footwear", "Health"])
            fig_state1, fig_comp_state1 = forecast_cpi_category.cpi_category(optionA)
            col3, col4 = st.columns(2)
            col3.pyplot(fig_state1)
            col4.pyplot(fig_comp_state1)

    elif task_selection == "Exchange Rates":

        optionA = st.selectbox("Select Currency:", ["USD", "SGD", "CAD"])
        if optionA:
            fig_state, fig_comp_state = forecast_exchangerate.exchange(optionA)
            
            st.pyplot(fig_state)
            st.pyplot(fig_comp_state)

    elif task_selection == "Malaysia GDP":

        optionA = st.selectbox("Select Series Type:", ["Real / SA", "Real", "Nominal"])    
        if optionA:
    
            fig_state, fig_comp_state = forecast_gdp.gdp(optionA)
            
            st.pyplot(fig_state)
            st.pyplot(fig_comp_state)

    elif task_selection == "Labour Stats":
        optionA = st.selectbox("Select Series Type:", ["Unemployment Rate", "Participation Rate", "Employment-Population Ratio"])  

        if optionA:
            fig_state, fig_comp_state = forecast_labor.labour()
            
            st.pyplot(fig_state)
            st.pyplot(fig_comp_state)

    elif task_selection == "Inflation Rate":

        fig_state, fig_comp_state = forecast_inflation.inflation()
        
        st.pyplot(fig_state)
        st.pyplot(fig_comp_state)

    
        
        


if __name__ == "__main__":
    main()
