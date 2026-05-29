import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

#page configuration and layout
st.set_page_config(page_title = "Retail Sales Dashboard", layout = "wide")
df = pd.read_csv("data/Sample - Superstore.csv", encoding='latin-1') # load the dataset
df['Order Date'] = pd.to_datetime(df['Order Date']) #convert Order Date to datetime format
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Quarter'] = df['Order Date'].dt.quarter
df['Shipping Duration'] = (df['Ship Date']-df['Order Date']).dt.days  #shipping Duration # ship date minus order date

model = joblib.load("models/best_model.pkl") # loading the saved model
st.title("Retail Business Intelligence & Sales Forecasting Dashboard")
st.markdown("This dashboard analyzes retail sales performance, profitability trends,customer segments, and demand forecasting using machine learning.")

#KPIs
st.markdown("------------") #hortizontal divider line
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
avg_order = df['Sales'].mean()
total_orders = df.shape[0]
#label = title of the card, value = the number to display
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Total Sales", value=f"${total_sales:,.0f}")
col2.metric(label="Total Profit", value=f"${total_profit:,.0f}")
col3.metric(label="Average Order Value", value=f"${avg_order:,.2f}")
col4.metric(label="Total Orders", value=f"{total_orders:,}")

# charts/sales analysis
st.markdown("---")
st.subheader("Sales Analysis")
st.markdown("Analysis to identify high performing business areas")
col1, col2 = st.columns(2)
#Sales by category
category_sales = df.groupby('Category')['Sales'].sum().reset_index()
category_sales = category_sales.sort_values('Sales', ascending=False)
fig1 = px.bar(category_sales, x='Category', y='Sales',title='Sales by Category',color='Category',text_auto='.2s')
col1.plotly_chart(fig1, use_container_width=True)
#Monthly sales Trend
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
fig2 = px.line(monthly_sales, x='Month', y='Sales',title='Monthly Sales Trend',markers=True)
fig2.update_xaxes(tickvals=list(range(1,13)),ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
col2.plotly_chart(fig2, use_container_width=True)
col3, col4 = st.columns(2)
#Profit by region
region_profit = df.groupby('Region')['Profit'].sum().reset_index()
fig3 = px.pie(region_profit, names='Region', values='Profit',title='Profit by Region',hole=0.3)
col3.plotly_chart(fig3, use_container_width=True)
#Top 10 products
top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
fig4 = px.bar(top_products, x='Sales', y='Product Name',orientation='h',title='Top 10 Products by Sales',text_auto='.2s')
fig4.update_layout(yaxis={'categoryorder':'total ascending'})
col4.plotly_chart(fig4, use_container_width=True)
#Discount vs profit
st.markdown("------")
st.subheader("Discount Impact Analysis")
fig5 = px.scatter(df, x = 'Discount', y = 'Profit', color = 'Category', title = "Discount vs Profit", hover_data = ['Sub-Category'])
fig5.add_hline(y=0, line_dash='dash', line_color='red',annotation_text='Break-even line')
st.plotly_chart(fig5, use_container_width=True)
#profit trend
st.markdown("---")
st.subheader("Profit Trend Over Time")
yearly_profit = df.groupby('Year')['Profit'].sum().reset_index()
fig6 = px.line(yearly_profit, x='Year', y='Profit',title='Yearly Profit Trend', markers=True)
fig6.update_xaxes(tickvals=yearly_profit['Year'].tolist(),ticktext=[str(y) for y in yearly_profit['Year'].tolist()])
fig6.update_layout(width=300,height=350,margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig6, use_container_width=True)
# segement alalysis
st.markdown("---")
st.subheader("Segment Analysis")
col1, col2 = st.columns(2)
segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()
fig7 = px.pie(segment_sales, names='Segment', values='Sales',title='Sales by Segment', hole=0.3)
col1.plotly_chart(fig7, use_container_width=True)
segment_profit = df.groupby('Segment')['Profit'].sum().reset_index()
fig8 = px.bar(segment_profit, x='Segment', y='Profit',title='Profit by Segment', color='Segment', text_auto='.2s')
col2.plotly_chart(fig8, use_container_width=True)

#Sales prediction
st.markdown("---")
st.subheader("Predicted Sales")
st.markdown("Adjust the inputs below to predict the sales values for an order")

col1, col2 = st.columns(2)
#dropsown for categorical inputs
ship_mode = col1.selectbox("Ship Mode", ["First Class", "Second Class", "Standard Class", "Same Day"])
segment = col1.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
region = col2.selectbox("Region", ["East", "West", "Central", "South"])
category = col2.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
col3,col4 = st.columns(2)
# sliders for numeric inputs
quantity = col3.slider("Quantity", 1, 14, 3)
discount = col4.slider("Discount", 0.0, 0.8, 0.1)
year = col3.selectbox("Year", [2015, 2016, 2017, 2018])
month = col4.slider("Month", 1, 12, 6)
quarter = col3.slider("Quarter", 1, 4, 2)
shipping_duration = col4.slider("Shipping Duration (days)", 0, 7, 3)

# encoding the user inputs the same way we encoded during training
# the model was trained on numbers not text so we convert manually
ship_mode_map = {"First Class": 0, "Second Class": 2, "Standard Class": 3, "Same Day": 1}
segment_map = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
region_map = {"Central": 0, "East": 1, "South": 2, "West": 3}
category_map = {"Furniture": 0, "Office Supplies": 1, "Technology": 2}
sub_category_default = 0

#predict button
if st.button("Predict Sales"):
    input_data = np.array([[ship_mode_map[ship_mode],segment_map[segment],region_map[region],category_map[category],
    sub_category_default,quantity,discount,year,month,quarter,shipping_duration]])
    # model predicts log(Sales) because we trained on log transformed target
    # expm1 reverses the log transform to get actual dollar value
    log_prediction = model.predict(input_data)
    predicted_sales = np.expm1(log_prediction[0])
    st.success(f"Predicted Sales: ${predicted_sales:,.2f}")
st.markdown("""Prediction generated using a trained **XGBoost Regression Model**
with a **log-transformed target variable**.""")
# model performance section
st.markdown("-----")
st.subheader("Model Performance Comparison")
st.markdown("3 models were trained and compared. Log transform was applied to improve R2 score.")
st.markdown("The following models were trained and evaluated to predict sales performance. XGBoost achieved the best predictive accuracy.")
performance_data = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R2 Before Log': [0.04, 0.19, 0.19],
    'R2 After Log': [0.15, 0.53, 0.54],
    'MAE': [1.25, 0.85, 0.84],
    'RMSE': [1.47, 1.09, 1.08]
}
perf_df = pd.DataFrame(performance_data)
st.dataframe(perf_df, use_container_width=True, hide_index=True )
fig_perf = px.bar(perf_df, x='Model', y=['R2 Before Log', 'R2 After Log'],title='R2 Score Comparison — Before vs After Log Transform',barmode='group',text_auto='.2f')
st.plotly_chart(fig_perf, use_container_width=True)

#Business Insights
st.markdown("---")
st.subheader("Business Insights")

with st.expander("Click to view full Business Insights"):
    st.markdown("""
    1. **Technology drives the most revenue** — despite having fewer products than
       Office Supplies, Technology category generates the highest total Sales.

    2. **Discounting is hurting profitability** — orders with discount above 0.4
       almost always result in negative profit. The current discounting strategy
       needs review.

    3. **West region is the most profitable** — while South has decent Sales volume,
       West region consistently delivers higher profit margins.

    4. **Sales peak in Q4** — clear seasonal demand spike from October to December,
       likely driven by year end purchasing cycles. Inventory should be planned accordingly.

    5. **Consumer segment generates the most Sales** — but Corporate segment
       should be explored for higher margin opportunities.

    6. **Some high Sales products have negative profit** — high revenue does not
       always mean high profit. Discount heavy products are selling well but
       losing money per order.

    7. **Log transforming Sales revealed heavy skewness** — majority of orders are
       low value but a few bulk orders skew the average significantly. Business
       should not plan inventory based on average order value alone.
    """)

#Download report
st.markdown("---")
st.subheader("Download Report")
report_df = df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Total_Orders=('Sales', 'count'),
    Avg_Discount=('Discount', 'mean')
).reset_index()

csv = report_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Category Summary Report",
    data=csv,
    file_name="retail_sales_report.csv",
    mime="text/csv"
)
