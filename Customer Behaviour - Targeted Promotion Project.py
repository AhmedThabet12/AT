Theme: Analyzing Customer Performance and Segmentation (Marketing)
Tasks

1. Analyze the customer behaviour data.

2. Derive actionable insights and provide strategic recommendations.

# Import the relevant packages for data preporcesing, manipulation and analysis

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import seaborn as sns 

# Reshape the Dataset into a DataFrame 

df = pd.read_csv(r"C:\Users\PC\Desktop\data.csv", encoding='ISO-8859-1')

## 1. Data Exploration and Preprocessing

### 1.1. Data Preprocessing

# Find out the number of rows and columns 

print(df.shape) # We have 8 columns along with 541,909 datapoints 

# Which columns do we have here?

print(df.columns) # We have now the columns' names

Since we have capitals letters in the columns naming, it will be better to make them small. In addition, we change the data type of CustomerID to string or object.

# Smalling the the columns nams characters 

df.columns = df.columns.str.lower()

# Changing the data type of the customerid column (Leaving the missing values out of any transformation, converting the floating values to integer and then to str)

df['customerid'] = df['customerid'].apply(lambda x: str(int(x)) if pd.notna(x) and x == int(x) else (str(x) if pd.notna(x) else x))

# As the invoice date is in a timestamp format, it ought to be changed to year-month and year formats 

df['invoicedate'] = pd.to_datetime(df['invoicedate']) # Change the date column to datetime date type 

df['year'] = df['invoicedate'].dt.year # Extract the year from the invoice date column 

df['year_month'] = df['invoicedate'].dt.to_period('M').astype(str) # Extract and month and year the invoicedate column and format them 'year_month'

print(df.head())

By that,: 
1. The data types of 'customerid' and 'invoicedate' columns are changed into datetime and string objects respectively.
2. Reformatting the 'customerid' into object (getting rid of the float format)
3. Creating new date-related columns, 'month-year' and 'year' columns extracted from the 'invoicedate' column for aggregation and grouping.
    
### 1.2. Data Exploration

#Check the data overview

print(df.describe())

***Primary Finding***: The data we deal with here is significantly right-skewed as there is a discrepancy between the mean is considerably larger than the median values in quantity and unitprice columns. 

Based on the above finding, there is around 135000 missing values from the customer ID column, which accounts for 24%  of the total count of rows.
    
133600 out of 135000 customerid missing values are attirbuted to UK as country (98% of the missing customerids are associated with UK) while the remaining 2% is divided among the other countries. 

# Exploring the countries associated with the missing values in the description column

missing_customers=df[df['description'].isna()].groupby('country').size().sort_values(ascending=False)

print(missing_customers.sum())

print(missing_customers)

100% of the missing values of the description columns is associated with the United Kingdom. 

# To find out about the duplicates 

duplicates = df[df.duplicated(keep=False)]

print(duplicates)

# Another Attempt to test how much the duplicates among the data set are by listing the relevant columns to narrow down the results

unique_columns = ['customerid', 'stockcode', 'invoiceno','description','quantity','unitprice','invoicedate','country']

duplicate_combination = df[df.duplicated(subset=unique_columns, keep=False)]

print(duplicate_combination)

# Clean the duplicate duplicates by removing the last occurence of the duplicates rows

df = df.drop_duplicates(keep='first') # Keep the first duplicate row 

### How I dealt with the duplicate values?

1. Return the list of duplicate rows.
   
2. Another test made to check the number of duplicates by ensuring uniqueness of rows across all columns to make sure of the actual count of duplicates:

- Having only the first 3 columns as unique columns (customerid,stockcode, invoiceno) returned more than 20K duplicates
  
- But adding the rest of the columns, it returned the same count of duplicates I got at the first place (10147) duplicates.
  
3. I remmoved the last occurence of each duplicate group keeping the last occurence along with the unique rows.

### 1.3.  Data Exploration

# Create a new revenue column (Unit price * Quantity)

df['revenue'] = df['quantity'] * df['unitprice']

print(df.columns)

# Calculating the correlation coefficient among  quantity, unit price and revenueto find out the magnitude and direction of relationship between them

relevant_data = df[['unitprice','quantity','revenue']] # Subsetting the relevant columns for the correlation testing

correlation_matrix = relevant_data.corr() # calculating the correlation matrix among the 3 elements

plt.figure(figsize=(8,6))

sns.heatmap(correlation_matrix,annot=True, cmap='Spectral', square=True, cbar=True, fmt='.2f', annot_kws={"size": 12}, linewidths=0.5)

plt.title ('Correlation Matrix', fontsize = 16, weight='bold',)

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

print(correlation_matrix.round(2))

Findings: 
1. The correlation coefficient between revenue and quantity sold is 0.89 (strong - positive correlation), whilst the correlation coefficient between revenue and Unit Price was less than -0.16 (negative weak correlation). It says that revenue is more sensitive to any change in price in comparison to qunatity.
2. The sales volume (quantity sold) is central to revenue generation. The sales volume changes will have a stronger impact on the amount of revenue generated rather than price change adjustments.
3. It is worth to look into how the correlation cofficient value by product for further analysis.
    
# Finding out the revenue flucutations across the time period 

monthly_revenue = df.groupby('year_month')['revenue'].sum().reset_index()

# Divide monthly revenue by million for readability and display

print("Monthly Revenue (in tens of millions):")

print(monthly_revenue['revenue'] / 10000000)


# Plotting the monthly revenue generated across the given time period

monthly_revenue = df.groupby('year_month')['revenue'].sum().reset_index() # Grouping the revenue data by month 

# Plotting the revenue line chart

plt.figure(figsize=(6,6))

x = monthly_revenue['year_month']  # Defining X-axis 

y = monthly_revenue['revenue'].round(0).astype(int)/1000000 # Defining Y-axis

revenue_mean = y.mean() # Revenue Mean Value defined

revenue_median = y.median() # Revenue Median Value defined

# Plotting the line chart 

plt.plot(x,y,color='green',alpha=0.7,marker='o',linestyle='solid',label='Monthly Revenue')

plt.axhline(y=revenue_mean, color='red', linestyle='--', label='Average Revenue')

plt.axhline(y=revenue_median, color='blue', linestyle='--', label='Median Revenue')

plt.title(f'Total Revenue Breakdown by Month',fontsize=14,weight='bold')

plt.xlabel('Year-Month',fontsize=12)

plt.ylabel('Monthly Revenue in Millions (EUR)',fontsize=12)

plt.xticks(rotation =45)

plt.ylim(0)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()

***Observation***:

1. The monthly total revenue was under the average and median revenue lines till April 2011.
   
2. In May 2011, the montly revenue landed on 720,000 EUR but still less than the average line and went static through August 2011.
     
3. Starting August 2011, the revenue boomed through Nov 2011 and dropped under the average line in Dec 2011.
   
4. The monthly revenue has been struggling at the first quarter of 2011 and the elevated with signficiant increasing rate in the 4th quarter of 2011.

5. Seasonality has an impact on the level of revenue generation. As moving time forward, the monthly revenue has been on raise, regardless of the customers or goods behind it.


# Another way to look into the data by providing the quantity breakdown  

monthly_quantity = df.groupby('year_month')['quantity'].sum().reset_index() #Grouping the quantity by month

# Plotting the revenue line chart

plt.figure(figsize=(6,6))

x = monthly_quantity['year_month'] # Defining X-axis

y = monthly_quantity['quantity'].round(0).astype(int)/1000 # Defining Y-axis

quantity_mean = y.mean() # Quantity Mean Value defined

quantity_median = y.median() # Quanity Median Value defined

# Plotting the line chart 

plt.plot(x,y,color='green',alpha=0.7,marker='o',linestyle='solid',label='Monthly Quantity')

plt.axhline(y=quantity_mean, color='red', linestyle='--', label='Average Quantity')

plt.axhline(y=quantity_median, color='blue', linestyle='--', label='Median Quantity')

plt.xlabel('Year-Month',fontsize=12)

plt.ylabel('Monthly Quantity in Thousands',fontsize=12)

plt.title(f'Total Quantity Breakdown by Month',fontsize=14, weight='bold')

plt.xticks(rotation =45)

plt.ylim(0)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()

***Observation***:

1. The Quantity change over time has the same pattern and seasonality has an impact to it.

2. It aligns with the correlation efficient value between revenue and quantity sold. 

# Another way to look into the data by providing the invoices breakdown by month   

monthly_quantity = df.groupby('year_month')['invoiceno'].nunique().reset_index() # Grouping the quantity by month

# Plotting the revenue line chart

plt.figure(figsize=(6,6))

x = monthly_quantity['year_month'] # Defining X-axis

y = monthly_quantity['invoiceno'] # Defining Y-axis

quantity_mean = y.mean() # Invoices Mean Value defined

quantity_median = y.median() # Invoices Median Value defined

# Plotting the line chart 

plt.plot(x,y,color='green',alpha=0.7,marker='o',linestyle='solid',label='Monthly Quantity')

plt.axhline(y=quantity_mean, color='red', linestyle='--', label='Average Quantity')

plt.axhline(y=quantity_median, color='blue', linestyle='--', label='Median Quantity')

plt.xlabel('Year-Month',fontsize=12)

plt.ylabel('Monthly Quantity in Thousands',fontsize=12)

plt.title(f'Total Quantity Breakdown by Month',fontsize=14, weight='bold')

plt.xticks(rotation =45)

plt.ylim(0)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()

***Observation***:
1. The median and average values are almost the same which shows a clear sign of central data.
2. The number of invoices has been increasing starting from May 2011 but felt under the average linebut jacked up again.
3. Seasonlity had the same impact as does on quantity sold and revenue generated. 

# Another way to look into the data by providing the invoices breakdown by month   

monthly_unitprice = df.groupby('year_month')['unitprice'].mean().reset_index()  # Averaging the unit price by month

# Plotting the revenue line chart

plt.figure(figsize=(6,6))

x = monthly_unitprice['year_month'] # define the x-axis

y = monthly_unitprice['unitprice'] # define the y-axis

unitprice_mean = y.mean() # Average price mean value 

unitprice_median = y.median() # Average price median value 

plt.plot(x,y,color='green',alpha=0.7,marker='o',linestyle='solid',label='Monthly Unit Price')

plt.axhline(y=unitprice_mean, color='red', linestyle='--', label='Average Unit Price')

plt.axhline(y=unitprice_median, color='blue', linestyle='--', label='Median Unit Price')

plt.title(f'Total Unit Price Breakdown by Month',fontsize=14,weight='bold')

plt.xlabel('Year-Month',fontsize=12)

plt.ylabel('Monthly Unit Price',fontsize=12)

plt.xticks(rotation =45)

plt.ylim(0)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()

***Observation***:
1. The Average price had a different pattern across the given time period.
2. The only time period where the average price was over the median and average lines, as same as the quantity, count of invoices and revenue, was May and June 2011.
3. During the time period, where the quantity, revenue and count of invoices were skyrocketing, the average price was below average.
4. Another proof of the negative weak correlation between revenue and total quantity sold. 

# Look into the number of products purchased by customers, excluding unknown customer IDs

customer_id_product_breakdown = df[df['customerid'] != 'unknown']['customerid'].value_counts()

print(customer_id_product_breakdown)

# 2. Customer Performance Analysis

### Business Question

**Customer Performance Analysis**: Provide an overview of existing and new customers, including their counts and growth rates. Analyze and compare revenue per customer for both existing and new customers. Highlight the performance of large order customers versus the overall customer base.

### Steps to Achieve This

1. **Categorize Customers**: Customers will be divided into two categories: New and Existing.

   **Logic**: 
   - If the first order date, regardless of the number of invoices, falls within the corresponding month, the customer ID will be labeled as "New."
   - If the first order date does not match the current month, the customer will be classified as "Existing." From that point forward, any subsequent orders placed in following months will result in the customer being labeled as "Existing."

2. **Analyze Customer Data**: I will focus the analysis on the count and growth rates of new and existing customers on a monthly basis, as well as calculate the average revenue for each customer.

# Identify the first order date per customer based on the minute they placed the order 

first_order = df.groupby('customerid')['invoicedate'].min().reset_index() # creating a DataFramme to pinpoint the first order per customer

first_order.columns = ['customerid','first_order_date'] # renaming the first order column in sake of noaming convenience 

print(first_order.head()) # Checking  th functionality of the datarame

# Merge the first date DataFrames with the original DataFrame

df = df.merge(first_order, on='customerid', how='left', suffixes=('', '_fd'))

print(df.head())

# Create a new column to label the customers with "New" and "Existing" based on the first order placed against the corresponding month 

df['customer_type'] = np.where(df['year_month'] == df['first_order_date'].dt.to_period('M'), 'New', 'Existing')

print(df.head())

# Creating the DataFrame to find out the count of the new and exisiting customers:

customer_counts = df.groupby(['year_month','customer_type'])['customerid'].nunique().reset_index()
customer_counts.columns = ['year_month','customer_type','customer_counts'] # Renaming the customeer_count column 


# Calculating the growth rate of the new and existing customers:

customer_counts_pivot_table = customer_counts.pivot_table(index='year_month',columns='customer_type',values='customer_counts').fillna(0)

customer_counts_pivot_table['existing_growth_rate'] = customer_counts_pivot_table['Existing'].pct_change() * 100

customer_counts_pivot_table['new_growth_rate'] = customer_counts_pivot_table['New'].pct_change() * 100


# In sake of consistency, I will filter out the data where the Dec. 2010 data. 

customer_counts_pivot_table = customer_counts_pivot_table[customer_counts_pivot_table.index >= '2011-01']


print(customer_counts_pivot_table)

# Plot the customer breakdown by type in absolute values and percentages

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

# Define the X and Y axes 

x = customer_counts_pivot_table.index # define the x-axis

y_new = customer_counts_pivot_table['New'] # define the y line 1

y_existing = customer_counts_pivot_table['Existing'] # define the y line 2

# Plotting the first graph of the new and existing customers linecharts 

ax1.plot(x,y_new,color='red',marker='o',linestyle='dashed',label='New Customers')

ax1.plot(x,y_existing,color='green',marker='o',linestyle='dashed',label='Existing Customer')

# Defining and Showing the first Plot

ax1.set_xlabel(f'Year - Month',fontsize=12)

ax1.set_ylabel(f'Customers Type Count',fontsize=12)

ax1.set_title('Customers Breakdown by Type', fontsize=14, fontweight='bold')

ax1.tick_params(axis='x',rotation =45)

ax1.grid(True)

ax1.legend()


# Second Subplot for the Growth Rates 


x = customer_counts_pivot_table.index

y_new_growth = customer_counts_pivot_table['new_growth_rate'] # define the y line 1

y_existing_growth = customer_counts_pivot_table['existing_growth_rate'] # define the y line 2

ax2.plot(x, y_new_growth, color='red',marker='o',linestyle='dashed',label='New Customers Growth Rate') # I could have divided by 100 

ax2.plot(x, y_existing_growth, color='green',marker='o',linestyle='dashed',label='Existing Customers Growth Rate')

ax2.set_xlabel(f'Year - Month',fontsize=12)

ax2.set_ylabel(f'Customer Type Growth',fontsize=12)

ax2.set_title('Customers Base Growth %', fontsize=14, fontweight='bold')

ax2.axhline(y=0, color='black', linestyle='dotted')

ax2.tick_params(axis='x',rotation =45)

ax2.grid(True)

ax2.legend()

ax2.yaxis.set_major_formatter(PercentFormatter()) 

plt.tight_layout()

plt.show()

# Plot the revenue generated by each customer group month by month

revenue_sum = df.groupby(['year_month','customer_type'])['revenue'].sum().reset_index()

revenue_sum.columns = ['year_month','customer_type','revenue_sum']

revenue_sum_pivot_table = revenue_sum.pivot_table(index='year_month',columns='customer_type',values='revenue_sum').fillna(0)

# In sake of consistency, I will filter out the data where the Dec. 2010 data. 

revenue_sum_pivot_table = revenue_sum_pivot_table[revenue_sum_pivot_table.index >= '2011-01']

print(revenue_sum_pivot_table)

# Plot the revenue breakdown by customer type

fig = plt.figure(figsize=(6,6))

# Define the X and Y axes 

x = revenue_sum_pivot_table.index # define the x-axis

y_new = revenue_sum_pivot_table['New']/1000000 # define the y line 1

y_existing = revenue_sum_pivot_table['Existing']/1000000 # define the y line 2

# Plotting the linechart  of the new and existing customers' revneue linecharts 

plt.plot(x,y_new,color='red',marker='o',linestyle='dashed',label='New Customers')

plt.plot(x,y_existing,color='green',marker='o',linestyle='dashed',label='Existing Customer')

# Defining the Plot

plt.xlabel(f'Year - Month',fontsize=12)

plt.ylabel(f'Customers Type Revenue in Millions',fontsize=12)

plt.title(f'Revenue Breakdown by Customer Type',fontsize=14,weight='bold')

plt.tick_params(axis='x',rotation =45)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()

# find out the revenue per customer for both customer type groups 


# merge the data between revenue and some columns from the customer_counts 

df_merged=pd.merge(revenue_sum, customer_counts,on=['year_month','customer_type'],how='left',suffixes=('_rs','_cc')) 

# divide the revenue sum by the count of customers 

df_merged['revenue_per_customer'] = df_merged['revenue_sum']/df_merged['customer_counts']

df_merged['revenue_per_customer'] = df_merged['revenue_per_customer'].round(3)

# pivot the data having the the month as index and the customer type group as column 

df_merged_pivot = df_merged.pivot_table(index='year_month',columns='customer_type',values=['customer_counts','revenue_sum','revenue_per_customer']).fillna(0)

df_merged_pivot = df_merged_pivot[df_merged_pivot.index >= '2011-01']

print(df_merged_pivot.head())

revenue_per_customer_new = df_merged_pivot[('revenue_per_customer', 'New')]

revenue_per_customer_existing = df_merged_pivot[('revenue_per_customer', 'Existing')]

# plot the bar chart

plt.figure(figsize=(10, 6))

bar_width = 0.30  # width of bars

x = range(len(revenue_per_customer_new))  # x locations for groups

plt.bar(x, revenue_per_customer_new, width=bar_width, label='New Customers', color='red', alpha=0.7)

plt.bar([p + bar_width for p in x], revenue_per_customer_existing, width=bar_width, label='Existing Customers', color='green', alpha=0.7)

plt.title('Revenue per Customer for New and Existing Customers', fontsize=14, weight='bold')

plt.xlabel('Year-Month', fontsize=12)

plt.ylabel('Revenue per Customer', fontsize=12)

plt.xticks([p + bar_width / 2 for p in x], df_merged_pivot.index, rotation=45)

plt.legend()  

plt.grid(True)



plt.tight_layout() 
plt.show()

***Observation***:

1. The average revenue per existing customer has been consistently and significantly higher than that of new customers over the first 11 months.

2. This indicates that existing customers are the foundation of the revenue generated.

### looking into the data and group them by revenue (The performance of the top group against the whole customer base): 

The idea here we want to find out which customer group contributes the most to the total revenue. 

1. Group the customer revenue data in the percentile groups.
   
2. Calculate the revenue contribution of each group.

3. Group the customer data also by the invoices by customer group.

4. Calculate the contribution of each group into the total count of invoices.

# Grouping the revenue data by the customer id

customer_revenue = df.groupby('customerid')['revenue'].sum().reset_index().sort_values(by='revenue',ascending=False)

# Create the revenue percentiles 

customer_revenue['revenue_percentile']=pd.qcut(customer_revenue['revenue'], q = 10, labels = False) + 1 # from 0- to 1-based indexing

#Grouping the data by percentiles 
customer_revenue_percentile = customer_revenue.groupby('revenue_percentile')['revenue'].agg(['mean','median','sum','count']).sort_values(by='revenue_percentile',ascending=False)

# Calculate the revenue contribution of each percentile bracket and the cumlative and relative frequency of it

customer_revenue_percentile['cumlative_revenue'] = customer_revenue_percentile['sum'].cumsum()

customer_revenue_percentile['sum'] = customer_revenue_percentile['sum'].round(2)

customer_revenue_percentile['mean'] = customer_revenue_percentile['mean'].round(2)

customer_revenue_percentile['median'] = customer_revenue_percentile['median'].round(2)

total_revenue = customer_revenue_percentile['sum'].sum()

customer_revenue_percentile['relative_percentage'] = customer_revenue_percentile['sum'] / total_revenue

customer_revenue_percentile['cumlative_percentage'] = customer_revenue_percentile['cumlative_revenue'] / total_revenue 

print(customer_revenue_percentile)


# Grouping the quantity / orders placed by customer id

orders_placed = df.groupby('customerid').agg({'invoiceno':'count'}).reset_index().sort_values(by='invoiceno',ascending=False)

orders_placed.columns = ['customerid','orders'] # Renaming the 2nd column for clarity 

# Creating the percentile column 
orders_placed['order_percentile'] = pd.qcut(orders_placed['orders'], q = 10, labels = False) + 1  # from 0- to 1-based indexing

# Gropuingthe customers_invoices data by the percentile bracket
orders_placed_percentile = orders_placed.groupby('order_percentile')['orders'].agg(['mean','median','sum','count']).sort_values(by='order_percentile',ascending=False) 
total_invoices_count = orders_placed['orders'].sum()

# # Calculate the orders placed by each percentile bracket and the cumlative and relative frequency of it

orders_placed_percentile['frequency_orders']= round((orders_placed_percentile['sum'] / total_invoices_count) * 100,2)
orders_placed_percentile['cumulative_orders'] = round(orders_placed_percentile['sum'].cumsum(),2)
orders_placed_percentile['order_cumulative_%'] = round((orders_placed_percentile['cumulative_orders'] / total_invoices_count) * 100,2)
# Median Value: The Middle range number of products bought by a customer in the relevant percentile
# Mean Value: Average number of products bought by a customer in the relevant percentile
# Count: the total count of customers belongs to a certain percentile bracket
# Sum Value: total number of products purchased by a customers in a certain percentile bracket. 

print(orders_placed_percentile)

### ***Observation***:
1. The discrepancy between the central tendency values (median and mean) for revenue is significantly higher compared to the order volume.
2. The top revenue percentile bracket (10th percentile) contributed 60% of the total revenue during this period. Adding the next two brackets brought the total revenue contribution to almost 83%.
3. Nearly 50% of the total orders were placed by 436 customers (those in the 10th percentile).
4. The bottom 5 percentile groups accounted for less than 5% of total revenue and approximately 4% of the total orders.
5. The number of customers per percentile group ranged between 417 and 457, which indicates that the count of customers per group did not play a big role in the revenue or orders placed breakdown.
6. The gap between the median and mean values increases notably in the top percentile brackets (8th, 9th, and 10th), as the percentile increases. This is further evidenced by the widening range of the percentile/standard deviation values.
7. Over 77% of the total orders were placed by customers in the top percentiles (Percentile Brackets = 8, 9, and 10).

***Does it mean that the top customers by revenue and total orders placed are the customers with the largest number of invoices?***

Another interesting aspect to find out whether the number of invoices are matter of interest to look into

# Group the unique invoices count by customer id

unique_invoices = df.groupby('customerid')['invoiceno'].nunique().reset_index().sort_values(by='invoiceno',ascending=False)

unique_invoices.columns = ['customerid','unique_invoices_count']

print(unique_invoices.head(10))

# Check the commonality level in terms of the customers between the list of the top customers in terms of unique invoices and total orders placed

def print_common_customers(unique_invoices, orders_placed, n_list):
    for n in n_list:
        common_customers = set(unique_invoices['customerid'].head(n)).intersection(set(orders_placed['customerid'].head(n)))
        print(f"Top {n} common customer IDs of unique invoices: ({len(common_customers)} common customer IDs of total orders placed)")
        print("--" * 50)  # Separator for clarity

# Example usage: check top 10, 30, and 50
n_list = [10, 20,30, 40,50]
print_common_customers(unique_invoices, orders_placed, n_list)

Observation:

I observed that the overlap between the top customers in both lists decreases as the customer sample size expands.

The big customers have the bigger basket size, even among the top 10% customers group.

It will be interesting to look into it through further analysis.

# 3. Customer Retention and Lifetime Value Analysis

### ***Customer Retention an Lifetime Value Analysis***:
- They are both measurements of customer loyalty. They are both Customer Experience (CX) metrics.  
- Loyal customers stay longer, spend more and are easier to serve. 
The two metrics are inextricably linked and positively and exponentially correlated.  (It increases rapidly as the retention rate increases). 
Both CLV and retention rate helps calculate the ROI of their experience programs. 
Metric to validate a loyaly program is return on investment. 
Traditional Brand owners and retailers encroach into the ecommerce channel. 
Digital engagement with customer provides companies with data to understand the customer behaviour and optimize their marketing and product development. 
Better understanding of the user experience and brand image. 
Effective and efficient marketing.
Tailoring offers to their context and needs. This can be achieved by drawing on customer-related metrics 
It is better to invest in lucrative customers in the long run. 

   
### ***Customer Retention Rate***:

***Defintion***: Measures how many customers can be retained in a given time period. 

Retention Rate = ((Customers at End of Period - New Customers) / Customers at the start of Period) * 100

***Retention Rate Calculation and Opertionalization***: 

1. determine the time period (monthly)
   
2. group the customers per month
 
3. calculate the total number of customers who placed an order in month x and month x - 1

4. Retained customer is a customer whose customerid exists in the list of customers who placed orders in month x and x - 1

5. Retention Rate = Retained Customers Count / Total Customers Count in the previous month. By that, the retention rate is calculated on a monthly basis.   

# Create the Monthly Customers Object

monthly_customers = df.groupby('year_month')['customerid'].nunique().reset_index()

monthly_customers.columns = ['year_month','total_customers'] # Rename the second column for clarity

# Create empty lists to store year_month and retention_rate for all months

all_year_months = []

all_retention_rates = []

# Create a DataFrame to store the monthly retention rates 

monthly_retention_rate = pd.DataFrame(columns=['year_month', 'retention_rate', 'previous_month_customers', 'retained_customers'])

# Loop to calculate retention rate for each month

for i in range(1, len(monthly_customers)):
    current_month = monthly_customers.iloc[i]
    previous_month = monthly_customers.iloc[i - 1]

    # Get current and previous month customer IDs
    current_month_customers = df[df['year_month'] == current_month['year_month']]['customerid'].unique()
    previous_month_customers = df[df['year_month'] == previous_month['year_month']]['customerid'].unique()

    # Calculate retained customers (common between current and previous month)
    retained_customers = len(set(current_month_customers).intersection(set(previous_month_customers)))

    # Get the number of customers in the previous month
    previous_month_customer_count = len(previous_month_customers)

    # Calculate the retention rate
    retention_rate_value = (retained_customers / previous_month_customer_count) * 100 if previous_month_customer_count > 0 else 0

    # Append the year_month and retention rate for the current month to the lists
    all_year_months.append(current_month['year_month'])
    all_retention_rates.append(retention_rate_value)

    # Create a temporary DataFrame for the current month
    temp_df = pd.DataFrame({
        'year_month': [current_month['year_month']],
        'retention_rate': [retention_rate_value],
        'previous_month_customers': [previous_month_customer_count],
        'retained_customers': [retained_customers]
    })

    # Concatenate the temporary DataFrame to the final DataFrame
    monthly_retention_rate = pd.concat([monthly_retention_rate, temp_df], ignore_index=True)

# Plot the retention rate line chart

plt.figure(figsize=(6,6))

x = all_year_months # define the x-axis data 

y = all_retention_rates # define the y-axis data 

plt.plot(x, y, color='orange', linestyle='dashed', marker='o', label='Monthly Retention Rate')

plt.title("Monthly Retention Rate", fontsize=16,weight='bold')

plt.xlabel('Year - Month', fontsize=12)

plt.ylabel('Retention Rate (%)', fontsize=12)

plt.xticks(rotation =45)

plt.ylim(0)

plt.legend()

plt.grid()

plt.tight_layout()

ax = plt.gca()  # Get current axis 

ax.yaxis.set_major_formatter(PercentFormatter())

plt.show()

# Print the DataFrame containing monthly retention data
print(monthly_retention_rate)

***Observation***:

1. The retention rate was fluctuating over time.
   
2. The count of the retained customers increases during the high season time.

3. The highest and lowest values of the retention rates resemble with the highest and lowest points of the revenue and quantity sold, which indicates the top customers have organically higher retention rate, it is yet to be tested. 
 

### ***Customer Lifetime Value (CLV)***:

***Definition***: It measures the how much worth the customers over a specific period of time. It is valued in EUR in this case.

***Operationalization and Calculation***:

***Customer Lifetime Value***= Customer Value * Customer Lifespan

***Customer Value***= Average Purchase Value * Purchase Frequency 

***Average Purchase Value***= Total Revenue / Number of Orders 

***Purchase Frequency***= Number of Purchases / Number of years 

***Customer Lifespan***= The time difference between the first and last orders placed by each customers and then divide it by 30 to round to months

#customer_lifetime_value = customer_value * customer_lifespan
#customer_value = Average Revenue (Average Purchase Value) * purchase_frequency
#average_purchase_value = total_revenue / orders_count
#purchase_frequency = total_orders / customer_lifetime 
#customer_life_span = years 

# Merge the customer_revenue unique_invoices customer_invoices objects to calculate the Customer Lifetime Value components

merged_df = customer_revenue.merge(unique_invoices,on = 'customerid', how = 'inner')

customer_data_merge = merged_df.merge(orders_placed, on = 'customerid' , how = 'inner',suffixes=('_df','_ci'))

customer_data_merge.columns = ['customerid','revenue','revenue_percentile','total_purchases','orders','order_percentile'] # Renaming for clarity


# Calculate the Average Revenue 

customer_data_merge['average_revenue'] = customer_data_merge['revenue'] /  customer_data_merge['total_purchases'] if previous_month_customer_count > 0 else 0

customer_data_merge['average_revenue'] = customer_data_merge['average_revenue'].round(2)

# Calculate the Customer Lifespan (Customer Lifetime in months)

customer_lifetime = df.groupby('customerid').agg({'invoicedate':['min','max']}).reset_index().sort_values(by='customerid',ascending=False)

customer_lifetime.columns = ['customerid','first_purchase','last_purchase'] # Renaming the columns for clarity 

customer_lifetime['customer_lifetime_months'] = round((customer_lifetime['last_purchase'] - customer_lifetime['first_purchase']).dt.days / 30,0).astype(int)

# Merge the customer lifespan data and the and customer data merge, subsetting the relevant columns

customer_data = pd.merge(customer_data_merge[['customerid','revenue','revenue_percentile','total_purchases','average_revenue']], customer_lifetime[['customerid', 'customer_lifetime_months']], on='customerid')

# Calculate the purchase frequency (total_purchases / customer lifespan)

customer_data['purchase_frequency'] = customer_data['total_purchases'] / customer_data['customer_lifetime_months']

customer_data['purchase_frequency'] = customer_data['purchase_frequency'].round(1)

# Calculate the Customer Lifetime Value (Average Purchase(Average Revenue) * Purchase Frequency * Customer Lifespan)

customer_data['clv'] = customer_data['average_revenue'] * customer_data['purchase_frequency'] * customer_data['customer_lifetime_months']

customer_data['clv'] = customer_data['clv'].round(2)

print(customer_data.head())

# Look into the Average retention period per customers groups on revenue basis

# Group the customer lifespan by the revenue percentile groups 

customer_data_percentile = customer_data.groupby('revenue_percentile')['customer_lifetime_months'].agg(['mean', 'median']).reset_index().sort_values(by='revenue_percentile', ascending=False)

customer_data_percentile['mean'] = customer_data_percentile['mean'].round(0).astype(int)

customer_data_percentile['median'] = customer_data_percentile['median'].round(0).astype(int)


customer_data_percentile.columns = ['Percentile','Average CL','Median CL'] # Renaming the Columns for clarity

print(customer_data_percentile)

# Look into the Average retention period per customers groups on revenue basis

customer_data_percentile = customer_data.groupby('revenue_percentile')['clv'].agg(['mean','median']).reset_index().sort_values(by='revenue_percentile', ascending=False)

customer_data_percentile['mean'] = customer_data_percentile['mean'].round(2).astype(int) 

customer_data_percentile['median'] = customer_data_percentile['median'].round(2).astype(int)

customer_data_percentile.columns=['Percentile','Average CLV','Median CLV']

print(customer_data_percentile)

In the top percentile group (10th percentile), there is a significant disparity between the averge and median customer lifetime values, which implies thhe following: 
1. The top 10% customers' data is right-skewed, skeweing the average upwards.
2. There is a few customers who drive the revenue "Big spenders".
3. The Average CLV of the top 10% group is more than the average CLV of the following percentile groups combined, especially that more than 49% of orders placed, 60% of revenue driven by the top percentile group.  
4. The average and median customer lifetime values among the following percentile groups are almost the same, no any huge skewness.
5. On the other hand, average and median customer lifespans (in months) is almost the same across all percentile groups proving that the revenue is mainyl drived by a specific number of customers.

# Plot Customer Lifespan Distribution (to find out the customers distribution among the customer lifespan period)

plt.figure(figsize=(6, 6))

sns.histplot(customer_data['customer_lifetime_months'], bins=30, kde=True, color='green')

plt.title('Customer Lifespan Distribution', fontsize=16,weight='bold')

plt.xlabel('Retention Period in Months', fontsize=12)

plt.ylabel('Customers per Customer Lifespan group', fontsize=12)

plt.grid(True)

plt.show()


***Observation***:
 
 1. 200 customers had a 12-month retention period. These customers had a consistent relationship across the whole year.
 2. Another 300 customers were retained customers for 10- and 11-month period.
 3. We had also more than 800 customers with a retention period between 6 and 8 months. 


# Plot to plot CLV vs Customer Lifespan

plt.figure(figsize=(8, 6))

plt.scatter(customer_data['customer_lifetime_months'], customer_data['clv'] /1000, alpha=0.9, color='skyblue')

plt.title('Customer Lifetime Value vs. Customer Lifespan', fontsize=14, weight='bold')

plt.xlabel('Retention Period (Months)', fontsize=12)

plt.ylabel('CLV in thousands(EUR)', fontsize=12)

plt.ylim(0)

plt.grid(True)

plt.show()

Observation:
There is a positive exponential relationship between Customer Lifetime Value (CLV) and Customer Lifespan (Retention Period).
Two customers exhibit a 12-month lifespan and have the highest CLVs, exceeding 400,000 EUR.
One customer falls within the CLV bracket of 100,000 - 150,000 EUR, maintaining a steady and consistent relationship.
In the 50,000 - 100,000 EUR CLV bracket, there are only three customers: two with a 12-month relationship and one with a 10-month relationship.
The majority of customers have a CLV between 0 and 400,000 EUR, regardless of their lifespan.
It is crucial to segment the data based on customer lifespan in relation to CLV, as it reflects the total revenue expected to be generated.

# 4. Customer Segmentation:

Based on the above findings, we will segement our customers on two dimensions: 

1. Customer Lifetime Value (CLV) and Customer Lifespan (Rentention Period)
   
2. Customer Lifetime Value will be divided into 5 groups.

This allows us to establish benchmarks on both dimensions for recommending customers for targeted promotions or customized offers.
   
customer_segmentation = customer_data.groupby(['clv','customer_lifetime_months'])['customerid'].nunique().reset_index().sort_values(by='clv',ascending=False)

# Group the customer ids into CLV range groups based on their CLV values

# 1. Define the CLV range groups and label them 

bins = [0,50000,100000,150000,200000,250000,float('inf')]
labels = ['(0-50k)','(50k-100k)','(100k-150k)','(150K-200K)','(200K-250K)','250K+']

# 2. Group the customers based on the defined CLV ranges 

customer_segmentation['clv_group'] = pd.cut(customer_segmentation['clv'],bins = bins, labels=labels)

#customer_segmentation_pivot = customer_segmentation.pivot_table(index='customer_lifetime_months')

customer_segmentation_pivot = customer_segmentation.pivot_table(index='customer_lifetime_months',columns='clv_group',values='customerid',aggfunc='count')

print(customer_segmentation_pivot)

# Focus on the big spenders, the top customers with the highest CLV, as they are the guiding star to follow:

# 1. Subset the customer data for customers with CLV greater than or equal to 75,000 EUR
high_clv_customers = customer_data[customer_data['clv'] >= 100_000][
    ['customerid', 'customer_lifetime_months', 'revenue_percentile', 'revenue', 'total_purchases', 'purchase_frequency', 'average_revenue', 'clv']]

# Merge with the country information from the original DataFrame
high_clv_customers_info = pd.merge(high_clv_customers, df[['customerid', 'country']], on='customerid', how='left')

# Drop duplicates to get unique customer information
high_clv_customer_info_unique = high_clv_customers_info.drop_duplicates(subset=['customerid'])

# Display the unique high CLV customer information
print(high_clv_customer_info_unique)


# Create a use case of the top 5 CLV customers calling them "Big Spenders"

big_spenders = high_clv_customer_info_unique.nlargest(5, 'clv')

big_spenders_merge = pd.merge(big_spenders[['customerid','total_purchases','revenue','clv']],df[['customerid','year_month']],on='customerid',how='left')


big_spenders_breakdown = big_spenders_merge.groupby(['year_month','customerid'])['revenue'].sum().reset_index()


big_spenders_breakdown_pivot = big_spenders_breakdown.pivot_table(index='year_month',columns='customerid',values='revenue',aggfunc='sum')

print(big_spenders_breakdown_pivot)

# Plot the revenue performance of the top 5 CLV customers 

sns.set(style='whitegrid')

plt.figure(figsize=(8,8))

sns.lineplot(data=big_spenders_breakdown, x='year_month', y=big_spenders_breakdown['revenue']/100000, hue='customerid', marker='o',linestyle='dotted')


plt.title(f"Big Spenders Revenue Over Time", fontsize = 16,weight='bold')
plt.xlabel('Year - Month', fontsize = 12)
plt.ylabel('Revenue in Hundred Thousands')
plt.xticks(rotation = 45)
plt.legend(title= 'Customer ID', bbox_to_anchor = (1,1), loc = 'upper left')
plt.tight_layout()
plt.ylim(0)
plt.show()

# Another use case focusing on the customes with longer retention periods

# 1. Subset the customer data for customers with long retention periods
long_retention_customers = customer_data[['customerid', 'customer_lifetime_months', 'revenue_percentile', 'revenue', 'total_purchases', 'purchase_frequency', 'average_revenue', 'clv']]

# Merge with the country information from the original DataFrame
long_retention_customers_merge = pd.merge(long_retention_customers, df[['customerid', 'country','year_month']], on='customerid', how='left')

long_retention_customers_merge_pivot = long_retention_customers_merge.pivot_table(index='year_month',columns='customer_lifetime_months',values='revenue',aggfunc='sum')

# Display the unique high CLV customer information
print(long_retention_customers_merge_pivot)


# Plotting the pivot table of average purchase frequency for long retention customers
plt.figure(figsize=(8, 8))

# Line plot for each customer lifetime month
for col in long_retention_customers_merge_pivot.columns:
    plt.plot(long_retention_customers_merge_pivot.index, long_retention_customers_merge_pivot[col], label=f'{col} Months',marker='o')

# Customize the plot
plt.title('Revenue Breakdown by retention group', fontsize=14,weight='bold')
plt.xlabel('Year-Month', fontsize=12)
plt.ylabel('Total Revenue grouped by customers lifespan', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title="Customer Lifetime (Months)", bbox_to_anchor = (1,1), loc = 'upper left')


# Adjust layout for better fit
plt.tight_layout()

# Show the plot
plt.show()


# Breakdown the purchase frequency by retention groups

# 1. Subset the customer data for customers with lon retention periods
long_retention_customers = customer_data[['customerid', 'customer_lifetime_months', 'revenue_percentile', 'revenue', 'total_purchases', 'purchase_frequency', 'average_revenue', 'clv']]

# Merge with the country information from the original DataFrame
long_retention_customers_merge = pd.merge(long_retention_customers, df[['customerid', 'country','year_month']], on='customerid', how='left')

long_retention_customers_merge_pivot = long_retention_customers_merge.pivot_table(index='year_month',columns='customer_lifetime_months',values='purchase_frequency',aggfunc='mean')

# Display the unique high CLV customer information
print(long_retention_customers_merge_pivot)


# Plotting the pivot table of average purchase frequency for long retention customers
plt.figure(figsize=(10, 6))

# Line plot for each customer lifetime month
for col in long_retention_customers_merge_pivot.columns:
    plt.plot(long_retention_customers_merge_pivot.index, long_retention_customers_merge_pivot[col], label=f'{col} Months',marker='o')

# Customize the plot
plt.title('Purchase Frequency Breakdown by retention group', fontsize=14,weight='bold')
plt.xlabel('Year-Month', fontsize=12)
plt.ylabel('Purchase Frequency grouped by customers lifespan', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.legend(title="Customer Lifetime (Months)", bbox_to_anchor = (1,1), loc = 'upper left')


# Adjust layout for better fit
plt.tight_layout()

# Show the plot
plt.show()

***Observations***:

1. From another angle, we can see the customers with a higher retention rate tend to bring more revenue as notcied in the graph, however, we see there was an organic surge of revenue between August and Nov 2011, which is related to the high-season impact.

2. The purchase frequency of the 12-month retention customers has been technically the highest, followed by the seasonal customers, with retention period between 1 and 3 months, as their purchases was between August and Nov 2011.  

grouped_data = customer_data.groupby('customer_lifetime_months').agg({'revenue': 'mean'}).reset_index()
correlation_coefficient = grouped_data['customer_lifetime_months'].corr(grouped_data['revenue'])

print(f"Correlation coefficient between retention group and revenue: {correlation_coefficient:.3f}")
Correlation coefficient between retention group and revenue: 0.857

grouped_data = customer_data.groupby('revenue_percentile').agg({'revenue': 'mean'}).reset_index()
correlation_coefficient = grouped_data['revenue_percentile'].corr(grouped_data['revenue'])

print(f"Correlation coefficient between retention group and revenue: {correlation_coefficient:.3f}")
Correlation coefficient between retention group and revenue: 0.785

grouped_data = customer_data.groupby('customer_lifetime_months').agg({'clv': 'mean'}).reset_index()
correlation_coefficient = grouped_data['customer_lifetime_months'].corr(grouped_data['clv'])

print(f"Correlation coefficient between retention period and Customer Lifetime Value: {correlation_coefficient:.3f}")
Correlation coefficient between retention period and Customer Lifetime Value: 0.747

grouped_data = customer_data.groupby('customer_lifetime_months').agg({'total_purchases': 'mean'}).reset_index()
correlation_coefficient = grouped_data['customer_lifetime_months'].corr(grouped_data['total_purchases'])

print(f"Correlation coefficient between retention period and total purchases: {correlation_coefficient:.3f}")
Correlation coefficient between retention period and total purchases: 0.837

Focus on retention should be central to the company's policy to generate more revenue in this case. From the previous correlation coefficient, it is shown how retention period would enable higher revenue generation, level of engagement (Orders placed and quantity purchases) and more valuable CLV.

1. The positive exponential relationship between Customer Lifetime Value (CLV) and Retention Period reveals two phases of business relationship development:

- Early Growth: Customers with shorter retention periods contribute minimal revenue during initial interactions, resulting in lower CLVs.
Accelerating Growth: As retention extends, engagement among remaining customers intensifies, with over 60% of revenue and 49% of orders coming from the top 10th percentile group, leading to significantly higher CLVs.

- Adopting a retention-focused strategy alongside customer segmentation will effectively target high-value customers based on revenue generation and retention duration.

2. Retention is central to business success. It has a compounding effect, continuously generating revenue while increasing average revenue over time exponentially.


3. Big spenders—top CLV customers—are showing increasing engagement. Their ticket size and purchase frequency are rising over time, exemplifying how longer retention periods lead to higher CLVs.

# ***5. Strategic Recommendations***:
## ***1. Customer Segmentation***

1. Customer segmentation (CLV vs. Retention period) showed that the retention improvement should be the key focus to generate revenue as the longer the retention period, the higher customer lifetime value potential is.
   
2. However, there are only six customers made purchases of more than 100K in the given time period, whose retention rate ranges between 10 and 12 months. So, the focus should be customers with higher than average revenue, number of orders placed and purchase frequency to retain them as long as possible. 

3.  More than 850 customers, whose retention period is more than 10 months, with a CLV less than 100K, which is a potential yet to be exploit through targeted promotions by upselling and cross-selling the products they purchased. 

## ***2. Retention Improvement***

1. The focus should be on customers with a retention period of more than 7 months, as these customers place at least one order per month. This presents a significant opportunity to increase their Customer Lifetime Value (CLV) through targeted promotions based on their purchase history.

2. High CLV customers should receive personalized offers designed to encourage upselling and cross-selling of products they are likely to purchase, maximizing their revenue contribution.

3. Implement a product subscription model: Analyze the order patterns of top-tier customers to offer them the option to subscribe to specific products for regular delivery at set intervals, enhancing convenience and encouraging long-term engagement.
   
# Personalized or Targeted Promotion should be provided the top CLV customers or Long retention rates (6 months or more)

# Get the max CLV value
max_clv = customer_data['clv'].max()

# Filter the customer data for the highest CLV
high_clv_customers = customer_data[customer_data['clv'] == max_clv][['customerid', 'customer_lifetime_months', 'revenue_percentile', 'revenue', 'total_purchases', 'purchase_frequency', 'average_revenue', 'clv']]

# Merge with product and invoice information
high_clv_customers_top_products = pd.merge(high_clv_customers, df[['year_month', 'customerid', 'description', 'invoiceno']], on='customerid', how='inner')

# Group by product description and year-month, count how often each product has been ordered
high_clv_customers_top_products_grouping = high_clv_customers_top_products.groupby(['description', 'year_month'])['invoiceno'].count().reset_index().sort_values(by='invoiceno', ascending=False)

high_clv_top_products = high_clv_customers_top_products.groupby('description')['invoiceno'].count().reset_index().sort_values(by='invoiceno', ascending=False)


high_clv_top_products.columns = ['Description','Order Frequency']
# Print the pivot table with products as rows and months as columns, showing the frequency of orders
#print(high_clv_customers_top_products_grouping)

print(high_clv_top_products.head(20)) # That shows us that our top CLV customer's basket has 721 products. They can be offered targeted promotions based on their top prodcuts 

# [Another Example] : Focus on Personalized or Targeted Promotions for Top CLV Customers or Long Retention Rates (6 months or more) 

# Filter customer data for those with retention periods of 6 months or more
long_retention_customers = customer_data[customer_data['customer_lifetime_months']==12]

# Get the customers with the highest CLV
high_clv_long_retention_customers = long_retention_customers[['customerid', 'customer_lifetime_months', 'revenue_percentile', 'revenue', 'total_purchases', 'purchase_frequency', 'average_revenue', 'clv']]

# Merge with product and invoice information to analyze purchasing patterns
long_retention_customers_top_products = pd.merge(high_clv_long_retention_customers, df[['year_month', 'customerid', 'description', 'invoiceno']], on='customerid', how='inner')

# Group by product description and year-month, count how often each product has been ordered
long_retention_customers_top_products_grouping = long_retention_customers_top_products.groupby(['description', 'year_month'])['invoiceno'].count().reset_index().sort_values(by='invoiceno', ascending=False)

# Aggregate total orders by product description
long_retention_top_products = long_retention_customers_top_products.groupby('description')['invoiceno'].nunique().reset_index().sort_values(by='invoiceno', ascending=False)

# Rename columns for clarity
long_retention_top_products.columns = ['Description', 'Order Frequency']

# Display the top 20 most frequently ordered products by these customers
print(long_retention_top_products.head(20))


# ***6. Projection of the Recommendation tool***

Based on the analysis, there are 3 elements to consider when designing the recommendation tool:

1. Retention Period (Positive direction).)
2. Level of Engagement (Positive direction).).
3. Total Revenue earned (Positive direction).



The Recommendation tool will identify the customers based on their customer lifetime values and their retention period. 

The customers will fall into 3 categories: 

1. ***Big Spenders***: Customers whose customer lifetime value more than 100,000 EUR on a monthly basis and their retention period is more than or equal to 6 months. 

2. ***Potential Customers***: Customers whose customer lifetime value more than 50,000 EUR on a monthl basis and their retention period is more than or queal to 6 months.

3. ***Seasonal Customers***: Customers whose customer lifetime value is more than 100,000 EUR on a monthly basis during the high season period (August - November).

# Objective: To develop a recommendation tool that evaluates customers' Customer Lifetime Value (CLV) and retention period (measured in months). Based on specific parameters, customers will be categorized, enabling the identification of targeted promotions tailored to each segment.  

recommendation_tool = pd.merge(customer_data[['clv','customer_lifetime_months','customerid']],df[['customerid','stockcode','revenue','year_month']],on='customerid',how='inner')

conditions = [
    (recommendation_tool['clv'] >= 100_000) & (recommendation_tool['customer_lifetime_months'] >= 6), # Big Spender Condition  
    (recommendation_tool['clv'] >= 50_000) & (recommendation_tool['clv'] < 100_000) & (recommendation_tool['customer_lifetime_months'] >= 6) # Potential Customer Condition   
    ]

choices = [
           'Big Customer',
           'Potential Customer']

promotions = [
    
    '5% discount on top 10 products',
    '5% discount on top 10 products for the next 6 months',
    ]

recommendation_tool['label'] = np.select(conditions,choices,default = 'Customer')

recommendation_tool['loyalty_program'] = np.select(conditions,promotions,default = 'not_eligible')

#print(recommendation_tool.head())
print(recommendation_tool)

# Count the unique customer IDs per label

unique_customers_count = recommendation_tool.groupby('label')['customerid'].nunique().reset_index(name='customers_count')

print(unique_customers_count)

big_customer_ids = recommendation_tool[recommendation_tool['label'] == 'Big Customer']['customerid'].unique()

seasonal_customer_ids = recommendation_tool[recommendation_tool['label'] == 'Seasonal Customer']['customerid'].unique()

seasonal_customer_ids_list = seasonal_customer_ids.tolist() 

bigcustomer_ids_list = big_customer_ids.tolist()

print("List of big customers by customer id", bigcustomer_ids_list)

Out of the 5000 customers listed in the dataset, we have only 23 customers eligible for targeted promotion:

20 potential customers are eligibe for 5% discount on their total products for the next 6 months.
15 seasonal customers, who we are looking to turn them to potential / big customers, are also eligible for targeted promotion to expand their retention period beyond the high-season period.
10 big customers, who are regular with big ticket sizes and we look to retain them, are also eligible for targeted promotion for 5%.

# Calculate the repeat purchase (Customers with multiple purchases / total number of customers)

purchase_repeat_customers = df.groupby('customerid')['invoiceno'].nunique().reset_index(name='total_purchases')

multiple_purchases_customers = purchase_repeat_customers[purchase_repeat_customers['total_purchases'] > 1]

num_multiple_purchases_customers = multiple_purchases_customers['customerid'].nunique()

total_customers = purchase_repeat_customers['customerid'].nunique()

repeat_purchase_rate = num_multiple_purchases_customers / total_customers

repeat_purchase_rate = round(repeat_purchase_rate,3) * 100

print(f"Repeat Purchase Rate: {repeat_purchase_rate}%")

Repeat Purchase Rate: 80.0%


## ***Findings***:

1. We have 10 big customers whose CL is longer than or equal to 6 months - CLV is more than 100,000 EUR.
   
2. We have another 15 potential customers whose CL is longer than 6 months.
   
3. More than 92% of the customers are customers during the high season.

4. A repeat purchase rate is 70% is an advantage should be exploited. As it indicates that 70% of our customers come back for another purchase at least. 

5. Targeted promotions and loyalty programs should aim for:

- Retaining big customers through discounts ensuring their ticket size gets bigger over time.
  
- Turning Potential customers to big customers focusing on increasing their enagagement rate.

- Seasonal customers are another potential customers class to consider for low season time. Further Analysis will be neded in this case.