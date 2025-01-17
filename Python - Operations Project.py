# ***Business Case***:

A company wants to find out the level of operational utilization in handling customer cases and how it would reflect on the customer experience. In addition, it is imperative to identify the bottlenecks in the processing workflow. 

The company seeks to optimize the operational setup and improve the overall customer experience in this case. 

### ***Task***: 

From Data perspective, we ought to pinpoint the bottlenecks in the operational workflow and how the customer experience was impacted by that. 

# *1. Data Exploration*

# Import the necessary Python packages 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# Read and reformat the file 

df = pd.read_csv(r"C:\Users\PC\Downloads\Operations case - Data - operational_analyst_homework_data.csv", encoding='ISO-8859-1')

# The DataFrame Shape

df.shape # We have 473,316 records and 11 columns

df.head()

df.dtypes

# Reformat the columns

df.columns = df.columns.str.lower()

print(df.columns)

# For convenience, the data types of case_id is changed

df['case_id'].astype(int).astype(str)

# changing the data type from object to datetime handling_time_start, handling_time_end and case_created_time

df['handling_time_start'] = pd.to_datetime(df['handling_time_start'])

df['handling_time_end'] = pd.to_datetime(df['handling_time_end'])

df['case_created_time'] = pd.to_datetime(df['case_created_time'])

# deleting the 'unnamed: 10' column

del df['unnamed: 10']


df.describe()

print(df['case_id'].nunique()) 

While we have more than 470,000 cases, the unique cases are 294000. That means 38% atmost of the cases were visited more than one time. 

# *2. Exploratory Data Analysis*

### 2.1. Exploring the data related to the cases 

# Count the cases by their type 

case_type_counts = df['case_type'].value_counts() # Cases breakdown by count

case_type_percentages = df['case_type'].value_counts(normalize=True) # Cases breakdown by percentage

case_type_pd = pd.DataFrame({"Counts in Thousands":round(case_type_counts/1000,2), "Percentage":round(case_type_percentages*100.0,2)})

print(case_type_pd)

**Observation**: Case Type 3 has the biggest share of the cases being handled in the given time period.

# Visualize the count of cases by month 

# Change the data type of the 'Handling Time Start' Column 

df['handling_time_start'] = pd.to_datetime(df['handling_time_start'])

# Extract year and month 

df['year'] = df['handling_time_start'].dt.year 

df['month'] = df['handling_time_start'].dt.month

df['year_month'] = df['handling_time_start'].dt.to_period('M').astype(str)

# Group by the year and month to count the cases 

case_counts = df.groupby(['year_month']).size().reset_index(name='count')

# Visualize the graph 

plt.figure(figsize=(5,5))

cases_in_thousands = case_counts['count'] / 1000

plt.plot(case_counts['year_month'], cases_in_thousands, marker='o', color='green', linestyle='--', alpha=0.8)

plt.title("Number of Cases started", fontsize=12)

plt.xlabel("Month", fontsize=10)

plt.ylabel("Cases Count in Thousands", fontsize=10)

plt.xticks(rotation=45)

plt.grid(alpha=0.4)

plt.show()

More than 77% of the cases handled were type 3 cases. 

***FINDING***: The discrepancy between the median and average handling time duration indicates that some cases may take more time than others

# Look into the data to find out the breakdown of cases by the final state

by_final_state = df.groupby(['final_state','case_type'])['case_id'].nunique().unstack().fillna(0).astype(int)


print(by_final_state)

Observation:

The absolute majority of the cases were resolved.
Resolved cases are cases that got either approved, closed, no review needed, or rejected.

# To check whether we have cases with more than one actor on: 

cases_actors = df.groupby('case_id')['actor_id'].nunique().reset_index(name='unique_actor_count')

# Filter the cases where there is more than one actor 

cases_with_multiple_actors = cases_actors[cases_actors['unique_actor_count'] > 1]

print(cases_with_multiple_actors.nunique()) 

83928 cases (20%) of the total cases were handled by multiple operations associates. 

# It is the nexus between customers and cases is one-to-many relationship 

customer_case = df.groupby('customer_id')['case_id'].size()

customer_case = customer_case.sort_values(ascending=False) 

customer_case = customer_case.reset_index(name='cases_count')

unique_customers_count = customer_case['customer_id'].nunique()

customers_multiple_cases = customer_case[customer_case['cases_count'] > 1]

customers_multiple_cases_count = customers_multiple_cases.nunique() 

print("Total Count of Unique Customers: ", unique_customers_count, "Customers")
print("Count of Revisiting Customers: ", customers_multiple_cases_count, "Customers")

### 2.2. Workload Rate

To estimate the overall operational health and as per the available data, Task-based Workload Rate is a metric to calculate the count of cases handled by an ops associate:

Task-Based Workload Rate = Count of Cases / Number of Employees

# To find out the workload rate over time 

# 1. extract the workload data related to number of Ops associates and cases grouping them by month 

workload_data = df.groupby('year_month').agg(unique_cases=('case_id','nunique'),unique_actors=('actor_id','nunique'))

# 2. Calculate the cases count change rate

workload_data['case_count_pct'] = workload_data['unique_cases'].pct_change().fillna(0) 

workload_data['case_count_pct'] = workload_data['case_count_pct'].round(2) * 100.0

# 3. Calculate the workload rate and change rate

workload_data['workload_rate'] = workload_data['unique_cases'] / workload_data['unique_actors'] # Dividing the count of cases by the number of ops associates

workload_data['workload_rate'] = workload_data['workload_rate'].round(2)

workload_data['workload_pct'] = workload_data['workload_rate'].pct_change().fillna(0)

workload_data['workload_pct'] = workload_data['workload_pct'].round(2) * 100.0

workload_data['headcount_pct'] = workload_data['unique_actors'].pct_change().fillna(0)

workload_data['headcount_pct'] = workload_data['headcount_pct'].round(2)

print(workload_data)

# Singular Scalar Average values of the headcount, cases count and average workload rate:


monthly_average_cases_count = workload_data['case_count_pct'].mean().round(2)
print("Average Cases Count Change rate: ", monthly_average_cases_count, "%")

workload_average = workload_data['workload_pct'].mean().round(2)
print("Workload Change rate: ", workload_average, "%")


# 3. Visualize the workload rate 

plt.figure(figsize=(3,2))
                   
plt.plot(workload_data.index, workload_data['workload_rate'], color='orange', alpha=0.8, marker='o',linestyle='--')
         
plt.title("Monthly Workload Rate", fontsize=12)
         
plt.xlabel("Month", fontsize=10)
         
plt.ylabel("Workload Rate", fontsize=10)
         
plt.xticks(rotation=45)
         
plt.grid(alpha=0.4)
         
plt.show()

***FINDING***: The workload rate has been decreasing over time especially starting from January 2022 as the number of cases were decreasing, while the number of operations associates were increasing over time, but no with the same rate. 

## 2.2. Process and Cycle Time

Process Time and Current Time are metrics to estimate the operational efficiency and delivery quality.

Process Time: The time difference between the start handling timestamp and the end handling timestamp.

Cycle Time: The time difference between the case creation timestamp and the end handling timestamp.

# 1. Reformat the timestamps 
df['handling_time_start'] = pd.to_datetime(df['handling_time_start'])

df['handling_time_end'] = pd.to_datetime(df['handling_time_end']) 

df['case_created_time'] = pd.to_datetime(df['case_created_time'])

df['process_time'] = df['handling_time_end'] - df['handling_time_start']

df['cycle_time'] = df['handling_time_end'] - df['case_created_time']

df['process_time'] = round(df['process_time'].dt.total_seconds()/60,2) # convert the duration to minutes format

df['cycle_time'] = round(df['cycle_time'].dt.total_seconds()/60,2) # Convert the duration to minutes format 

# 2. Subsetting the relevant data (Handling and case creation timestamps)

cycle_time_data = df[['year_month','case_type','case_id','process_time','cycle_time','case_created_time']] 

# 3. Calculate the process time and cycle time (in Min) and group them by month

cycle_time_data_grouped = cycle_time_data.groupby('year_month').agg(cases_count=('case_id','count'),process_dur=('process_time','sum'),cycle_dur=('cycle_time','sum'))

cycle_time_data_grouped['process_time_in_min'] = cycle_time_data_grouped['process_dur'] / cycle_time_data_grouped['cases_count']

cycle_time_data_grouped['cycle_time_in_min'] = cycle_time_data_grouped['cycle_dur'] / cycle_time_data_grouped['cases_count']

cycle_time_data_grouped['process_time_in_min'] = cycle_time_data_grouped['process_time_in_min'].round(2)

cycle_time_data_grouped['cycle_time_in_min'] = cycle_time_data_grouped['cycle_time_in_min'].round(2)

cycle_time_data_grouped['creation_handling_difference_in_min'] = cycle_time_data_grouped['cycle_time_in_min'] - cycle_time_data_grouped['process_time_in_min']

cycle_time_data_grouped['creation_handling_difference_in_min'] = cycle_time_data_grouped['creation_handling_difference_in_min'].round(2)

# Reshape the results into a DataFrame

cycle_time_data_grouped = cycle_time_data_grouped.reset_index()

cycle_time_data_grouped[['year_month', 'cases_count', 'process_time_in_min', 'cycle_time_in_min', 'creation_handling_difference_in_min']]

*Observation*: 
1. The discrepancy between the cycle time and process time indicates to the inefficiency in terms of workflow.
2. A room for improvment in terms of operational efficiency, especially the case assignment and handling.

# Plot the Process Time graph 

plt.figure(figsize=(3,2))

plt.plot(cycle_time_data_grouped['year_month'], cycle_time_data_grouped['process_time_in_min'], marker='o', linestyle='--', color='green', alpha=0.8)

plt.title('Process Time', fontsize=12)

plt.xlabel('Month', fontsize=10)

plt.ylabel('Process Time (in Min)', fontsize=10)

plt.xticks(rotation=45)

plt.grid(alpha=0.4)

plt.show()

# Plot the Process Time graph 

plt.figure(figsize=(3,2))

plt.plot(cycle_time_data_grouped['year_month'], cycle_time_data_grouped['cycle_time_in_min'], marker='o', linestyle='--', color='green', alpha=0.8)

plt.title('Cycle Time', fontsize=12)

plt.xlabel('Month', fontsize=10)

plt.ylabel('Cycle Time (in Min)', fontsize=10)

plt.xticks(rotation=45)

plt.grid(alpha=0.4)

plt.show()

# Calculate the average and median Process Time, Cycle Time and time difference between case creation and processing

average_cycle_time = cycle_time_data_grouped['creation_handling_difference_in_min'].mean().round(2)

average_process_time = cycle_time_data_grouped['process_time_in_min'].mean().round(2)

average_creation_handling_time = cycle_time_data_grouped['cycle_time_in_min'].mean().round(2)

print("Average Cycle Time: ", average_cycle_time, " min")

print("Average Process Time", average_process_time, " min")

print("Average time difference between case creation and heandling: ", average_creation_handling_time, " min")


# Test the correlation between the process time, cycle time and cases count

cycle_time_correlation_data = cycle_time_data_grouped[['cycle_time_in_min','process_time_in_min','cases_count']] 

correlation_matrix = cycle_time_correlation_data.corr()

plt.figure(figsize=(4,4))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.05)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.title('Correlation Heatmap')

plt.show()

***Finding***: There is a positive moderate correlation of 0.54 between cases count and process time and cycle time.  


# Calculate the cycle time and process time temporally 

cycle_process_time_month = df.groupby('year_month').agg(cases_count=('case_id','size'),cycle_time=('cycle_time','mean'),process_time=('process_time','mean')).reset_index()

cycle_process_time_month['process_time'] = cycle_process_time_month['process_time'].round(2)

cycle_process_time_month['cycle_time'] = cycle_process_time_month['cycle_time'].round(2)

cycle_process_time_month


ops_decision_type = df.groupby('ops_decision').agg(cases_count=('case_id','size'),cycle_time=('cycle_time','mean'),process_time=('process_time','mean')).reset_index()

ops_decision_type['cycle_time'] = ops_decision_type['cycle_time'].round(2)

ops_decision_type['process_time'] = ops_decision_type['process_time'].round()

ops_decision_type


# Look into the cases' cycle and process time based on the case type 

case_type_time = df.groupby('case_type').agg(cases_count=('case_id','size'),cycle_time=('cycle_time','mean'),process_time=('process_time','mean')).reset_index()

case_type_time['cycle_time'] = case_type_time['cycle_time'].round(2)

case_type_time['process_time'] = case_type_time['process_time'].round(2) 

case_type_time


Observations:

By Case Type: Case Type 2 cases had the longest cycle time, however, Case Type 3 cases had the longest process time due to the share of type 3 cases in the totaeaml cases count.
By Month: The month with the longest cycle time was Feb'22, 3500 minutes per case and the longest process time was in Apr'22.
By Final State: The cases which are subject to reviewing had the second longest cycle time of 16000 minutes (11 days).
By Ops decision: The cases that were unassigned or the handling team changed had the longest cycle time of 19000 as they were left unassigned, or being assigned to a different team.

# We look into the customers and their cases 

customer_case_data = df.groupby('customer_id')['case_id'].nunique()

maximum_customer_case = customer_case_data.max() 

print(maximum_customer_case)

# Look into the cases' cycle and process time based on their final state 

final_state_time = df.groupby('final_state').agg(cases_count=('case_id','size'),cycle_time=('cycle_time','mean'),process_time=('process_time','mean')).reset_index()

final_state_time['cycle_time'] = final_state_time['cycle_time'].round(2)

final_state_time['process_time'] = final_state_time['process_time'].round(2)

final_state_time


cycle_time_case_type = cycle_time_data.groupby('case_type').agg(cases_count=('case_id','count'),process_dur=('process_time','sum'),cycle_dur=('cycle_time','sum'))

cycle_time_case_type['process_time_in_min'] = cycle_time_case_type['process_dur'] / cycle_time_case_type['cases_count']

cycle_time_case_type['cycle_time_in_min'] = cycle_time_case_type['cycle_dur'] / cycle_time_case_type['cases_count']

cycle_time_case_type['process_time_in_min'] = cycle_time_case_type['process_time_in_min'].round(2)

cycle_time_case_type['cycle_time_in_min'] = cycle_time_case_type['cycle_time_in_min'].round(2)

print(cycle_time_case_type)

However the type two cases were the least, the cycle time is the longest. 
the type three cases are the most cases handled take the longest time to handle. 
Type one cases are the easiest in terms of handling. 

## 3. Customer Experience 

### 3.1. Resolution Rate

As the focus here will be on quality and speed delivery, we ought to us the resolution rate to guage the customer experience in terms of quality. 

Resolution Rate Calc = (resolved_cases / total_cases_handled) * 100

Resolved cases: Cases where handled and had a final state (approved, rejected, closed, or no ops review needed)

The goal here is to find out how the resolution rate was changing over time and whether it was subject to other factors

# Find the different final states available in the dataset

ops_decision_items = df['final_state'].unique()

print(ops_decision_items)

# Define the resolved cases

resolved_states = ('APPROVED', 'CLOSED', 'NO_REVIEW_NEEDED', 'REJECTED')

# Apply Lambda fuction to calculate the resolution rate on a monthly basis 

monthly_resolution_rate = df.groupby('year_month').apply (
    lambda group: round (
        group[group['final_state'].isin(resolved_states)]['case_id'].nunique() / group['case_id'].nunique() * 100.0
        ,2)   
)

# To rename the monthly resolution rate, convert the series into a DataFrame

monthly_resolution_rate = monthly_resolution_rate.reset_index(name='monthly_resolution_rate')

print(monthly_resolution_rate)

# Calculate the median and average resolution rate 

average_resolution_rate = monthly_resolution_rate['monthly_resolution_rate'].mean() # Average Resolution Rate across the given time period 

median_resolution_rate = monthly_resolution_rate['monthly_resolution_rate'].median() # Median Resolution Rate across the given time period 

print("Average Resolution Rate: ", average_resolution_rate, "%")

print("Median Resolution Rate: ", median_resolution_rate, "%")

Average Resolution Rate:  96.33 %
Median Resolution Rate:  98.09 %



duplicate_cases = df.duplicated(subset=['case_id', 'year_month']).sum()

print("Number of duplicate cases:", duplicate_cases)

Number of duplicate cases: 171839

# Visualize the MoM resolution Rate 

import matplotlib.ticker as ticker

plt.figure(figsize=(3,2))

plt.plot(monthly_resolution_rate['year_month'], monthly_resolution_rate['monthly_resolution_rate'], color='orange', alpha=0.8, marker='o',linestyle='--')

plt.title("Resolution Rate", fontsize=12)

plt.ylim(0,100)

plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

plt.xlabel("Month", fontsize=10)

plt.ylabel("Resolution Rate", fontsize=10)

plt.xticks(rotation=45)

plt.grid(alpha=0.4)

plt.show()

# Count of cases per customer

case_counts = df['case_id'].value_counts()

print(case_counts[case_counts > 1])  # Show cases handled multiple times

**First Pay Resolution Rate (FPRR)**: It is the rate of the cases got resolved from the first time.

**First Pay Resolution Rate (FPRR)** = count of first resolved cases / count of resolved cases. 

**Steps**:

1. Sort the values by the handling start time to order the cases by the first data of handling.
2. Filter out the cases where the case was still still under review, unassigned or to be assigned to another team.
3. Keep the cases with the first handling start time.
4. Divide the first resolved cases (Cases being resolved on the first attempt) by the total resolved cases.


sorted_df = df.sort_values(by=['handling_time_start'])

resolved_ops_decision = ('OPS_APPROVE','OPS_CLOSE','OPS_NO_REVIEW_NEEDED','OPS_REJECT')

resolved_states = ('APPROVED', 'CLOSED', 'NO_REVIEW_NEEDED', 'REJECTED')


first_pass_resolution_rate = sorted_df.groupby('year_month').apply (
    lambda sorted_df: round (
        sorted_df[sorted_df['ops_decision'].isin(resolved_ops_decision)].drop_duplicates(subset='case_id', keep='first')['case_id'].nunique() / sorted_df[sorted_df['final_state'].isin(resolved_states)]['case_id'].nunique() * 100.0
        ,2)   
)

first_pass_resolution_rate = first_pass_resolution_rate.reset_index(name='FPRR')

print(first_pass_resolution_rate)

# Average and Median First Pass Resolution Rate to find out whether there were any outlier

average_FPRR = first_pass_resolution_rate['FPRR'].mean().round(2) # Average FPRR

print("Average First Pass Resolution Rate: ", average_FPRR, "%")

median_FPRR = first_pass_resolution_rate['FPRR'].median() # Median FPRR

print("Median First Pass Resolution Rate: ", median_FPRR, "%")

Average First Pass Resolution Rate:  74.24 %
Median First Pass Resolution Rate:  73.23 %

Observation: 1 percent difference between average and median First Pass Resolution Rate shows that the percentage of cases being resolved from the first pass was stable across the given time period

# Visualize the FPRR

plt.figure(figsize=(3,2))

plt.plot(first_pass_resolution_rate['year_month'], first_pass_resolution_rate['FPRR'], color='green', alpha=0.8, marker='o',linestyle='--')

plt.title("First Pass Resolution Rate", fontsize=12)

plt.ylim(0,100)

plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

plt.xlabel("Month", fontsize=10)

plt.ylabel("First Pass Resolution Rate", fontsize=10)

plt.xticks(rotation=45)

plt.grid(alpha=0.4)

plt.show()

# Alternative: Calculate the Follow-up rate  

case_duplicates = df['case_id'].value_counts()

actual_duplicates = case_duplicates[case_duplicates > 1] 

actual_duplicates_count = actual_duplicates.count()

total_unique_cases = df['case_id'].nunique()

follow_up_rate = actual_duplicates_count / total_unique_cases * 100.0

print("Total Duplicate Cases: ", actual_duplicates_count, "cases")

print("Total Unique Cases", total_unique_cases, "cases")

print("Follow-up rate: ", round(follow_up_rate,2), "%")

Total Duplicate Cases:  102503 cases
Total Unique Cases 294292 cases
Follow-up rate:  34.83 %

**Case Processing Rate**: It is the count of processing per case 

# Find the processings count per case

processing_per_case = df.groupby('case_id')['processing_id'].size()

average_processing_per_case = processing_per_case.mean().round(2)

# Find the average processing count for revisited case 

revisited_processing_per_case = processing_per_case[processing_per_case > 1]

average_revisited_processing_per_case = revisited_processing_per_case.mean().round(2)

print("Average Processing per case: ", average_processing_per_case,"processings per case")

print("Average Processing per revisited case: ", average_revisited_processing_per_case, "Processings per case")

# Look into the Average Processing per revisited case by case type 

# 1. Merge the data

revisited_cases_df = df[df['case_id'].isin(revisited_processing_per_case.index)]


average_processing_by_case_type = (
    revisited_cases_df.groupby('case_type')['case_id']
    .apply(lambda x: revisited_processing_per_case.loc[x].mean()
           .round(2)
    )
)


print(average_processing_by_case_type)


Average Processing per case:  1.61 processings per case
Average Processing per revisited case:  2.75 Processings per case
case_type
TYPE_ONE       2.65
TYPE_THREE     3.54
TYPE_TWO      18.92
Name: case_id, dtype: float64

# To better understand the case processing and handling workflow 

discover_data = df[['case_id','ops_decision','final_state']]

discover_data_pivot = discover_data.pivot_table(index='final_state', columns='ops_decision', aggfunc='size').fillna(0).astype(int)

print(discover_data_pivot)

The Operations associate has the final authority on any case. Any case that is approved or rejected on the first attempt remains unchanged. 
**Count of cases per customer**: It is the count of cases per customer to find out how frequent the customers ask for support with a case

# Customers' case count 

customer_case_count = df.groupby(['customer_id','case_type'])['case_id'].size().reset_index(name='count') # Count the cases grouping by customers and case type

customer_case_count_pivot = customer_case_count.pivot_table(index='customer_id',columns='case_type',values='count').fillna(0).astype(int)

customer_case_count_pivot['Total'] = customer_case_count_pivot.sum(axis=1)

customer_case_count_pivot_sorted = customer_case_count_pivot.sort_values(by='Total', ascending=False)

print(customer_case_count_pivot_sorted)

# Find out the employees' functionality 

ops_assoc_data = df.groupby('actor_id')['case_id'].nunique()

ops_assoc_data = ops_assoc_data.sort_values(ascending=False)

total_cases = df['case_id'].nunique()



ops_assoc_data = ops_assoc_data.reset_index(name='cases_handled_assoc')

ops_assoc_data['relative_frequency'] = round(ops_assoc_data['cases_handled_assoc'] / total_cases,2)

ops_assoc_data_top_20 = ops_assoc_data.head(20) 

print (ops_assoc_data_top_20)

# *4. Employee Turnover and Turnover Rate*

# Group by month and collect unique active employees

active_employees = df.groupby('year_month')['actor_id'].unique().reset_index()
active_employees.columns = ['year_month', 'active_actor_ids']

# Add a column for the number of active employees
active_employees['active_count'] = active_employees['active_actor_ids'].apply(len)

# Calculate employees who left
active_employees['left_count'] = active_employees['active_actor_ids'].shift(1).combine(
    active_employees['active_actor_ids'], 
    lambda prev, curr: len(set(prev) - set(curr)) if prev is not None else 0
)

# Calculate the average active employees
active_employees['average_active'] = (
    active_employees['active_count'].shift(1) + active_employees['active_count']
) / 2

# Calculate the new joiners
active_employees['new_joiners'] = active_employees['active_actor_ids'].shift(1).combine(
    active_employees['active_actor_ids'],
    lambda prev, curr: len(set(curr) - set(prev)) if prev is not None else len(curr)
)

# Calculate the turnover rate
active_employees['turnover_rate'] = (
    active_employees['left_count'] / active_employees['average_active']
) * 100

# Display the result
print(active_employees[['year_month', 'active_count', 'left_count','new_joiners', 'average_active', 'turnover_rate']])

average_turnover_rate = active_employees['turnover_rate'].mean().round(2)
print(average_turnover_rate)
10.59

import matplotlib.pyplot as plt
import numpy as np

# Example DataFrame (replace with your actual data)
# active_employees = your DataFrame

# Define bar width and x positions
bar_width = 0.25
x = np.arange(len(active_employees['year_month']))  # Numeric positions for x-axis

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(8,6))

# Plot active count, left count, and new joiners side-by-side
ax1.bar(x - bar_width, active_employees['active_count'], color='green', width=bar_width, label='Active Employee Count', alpha=0.8)
ax1.bar(x, active_employees['left_count'], color='red', width=bar_width, label='Left Employees Count', alpha=0.8)
ax1.bar(x + bar_width, active_employees['new_joiners'], color='skyblue', width=bar_width, label='New Joiners Count', alpha=0.8)

# Set x-axis labels to correspond to year_month
ax1.set_xticks(x)
ax1.set_xticklabels(active_employees['year_month'], rotation=45)

# Labels and legend
ax1.set_xlabel('Month')
ax1.set_ylabel('Count of Employees')
ax1.tick_params(axis='y')


# Plot turnover rate on the second y-axis
ax2 = ax1.twinx()
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
ax2.plot(x, active_employees['turnover_rate'], color='purple', marker='o', linestyle='--', label='Turnover Rate (%)')
ax2.set_ylabel('Turnover Rate (%)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')


handles, labels = ax1.get_legend_handles_labels()  # Get handles and labels from ax1
handles2, labels2 = ax2.get_legend_handles_labels()  # Get handles and labels from ax2
handles.extend(handles2)  # Combine handles
labels.extend(labels2)  # Combine labels

ax1.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.2, 1.2))


# Title and Layout
plt.title('Employees Turnover and Turnover Rate')
fig.tight_layout()
plt.show()


case_type_data = df.groupby(['case_type', 'final_state'])['case_id'].size().fillna(0).astype(int).unstack()

print(case_type_data)

efficiency_score = df.groupby('year_month').agg(process_time=('process_time','mean'), cases_count=('case_id','nunique'))

efficiency_score['averge_time_allocated'] = round(efficiency_score['process_time'] * efficiency_score['cases_count'],2)

monthly_total_working_min = 40 * 4 * 60 

efficiency_score['efficiency_rate'] = round((efficiency_score['averge_time_allocated'] / monthly_total_working_min),2)

total_cases = df['case_id'].nunique()

efficiency_score['relative_efficiency'] = round((efficiency_score['averge_time_allocated'] / total_cases),2)

print(efficiency_score)

# 5. Efficiency Score (Hypothetical)

- The Efficiency Score will assess the level of efficacy of the operations setup in terms of speed and quality. 

- To do so, we have to set one key assupmtion:

1. All Operations associates were full time employees.
2. All associated employees are not on any vacation.  

- Efficiency Score will be entity-based metric but it can be aggregated on a deeper levels, like: Case Type, Actor ID, etc.

- Efficiency Score will be a lagging indicator assessing the ops performance.

- Efficiency Score ranges between 0 (Full Underutilized) and 100 (Full Utilized).

**Efficiency Score** = Total Process Time / Available Capacity 

Total Process Time = Average Process Time * Total Count of cases handled 

Avaiable Capacity = Business days count * current headcount * 480 ( Working minutes per business day)  

efficiency_data = df[['year_month', 'process_time', 'case_id', 'actor_id']] # Subsetting the relevant columns and records

aggregation = {'process_time':'mean', 'case_id': 'size', 'actor_id': 'nunique'} # Add the aggregated elements as a dictionary 

efficiency_data_grouped = efficiency_data.groupby('year_month').agg(aggregation) # grouping them by year_month 

efficiency_data_grouped.columns = ['avg_process_time', 'total_cases', 'headcount'] # Rename the columns for convenience 

efficiency_data_grouped.index = pd.to_datetime(efficiency_data_grouped.index, format = '%Y-%m') # Reformat the year_month column from string to datetime index

def count_business_days(month): # Define a function to count the business days as per the datetime index (year_month)
    start_date = month
    end_date = month + pd.offsets.MonthEnd(0)
    business_days = pd.bdate_range(start_date, end_date).size
    return business_days

efficiency_data_grouped['business_days'] = efficiency_data_grouped.index.map(count_business_days)

efficiency_data_grouped['total_process_time'] = efficiency_data_grouped['avg_process_time'] * efficiency_data_grouped['total_cases'] 

efficiency_data_grouped['available_capacity'] = efficiency_data_grouped['business_days'] * efficiency_data_grouped['headcount'] * 480

efficiency_data_grouped['efficiency_score'] = round((efficiency_data_grouped['total_process_time'] / efficiency_data_grouped['available_capacity'] * 100.0),2)

efficiency_data_grouped.columns = ['Average Process Time', 'Total Cases', 'Headcount', 'Business Days', 'Total Process Time', 'Available Capacity', 'Efficiency Score'] # Rename the columns for convenience 

efficiency_data_grouped.head()

The Key finding from the presented data that the efficiency score was increasing, indicating improved utilization over time in terms of processing speed and quality. However, there is still room for improvemet in overall utilization.  
