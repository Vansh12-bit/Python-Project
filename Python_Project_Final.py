import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv("C:/Users/ASUS/Documents/final_dataset_python.csv")
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())


#Objective No.1
# Summary statistics for Speed_limit and Number_of_Vehicles
summary_stats = df.describe()
# Identify the most common types of accidents and their severity
most_common_accidents = df["Accident_Severity"].value_counts()

#Objective No.2
# (2.1): Bar chart for Speed_limit and Number_of_Vehicles
summary_stats.loc["mean", ["Speed_limit", "Number_of_Vehicles"]].plot(kind='bar', color=['blue', 'red'])
plt.title("Average Speed Limit & Number of Vehicles in Accidents")
plt.xlabel("Category")
plt.ylabel("Average Value")
plt.show()

# (2.2): Pie chart for most common accident severities
plt.figure(figsize=(6, 6))
most_common_accidents.plot(kind='pie', autopct='%1.1f%%', colors=['green', 'orange', 'purple'])
plt.title("Distribution of Accident Severities")
plt.ylabel("")  # Hide default ylabel for better visualization
plt.show()

# (2.3): Histogram for accidents per day of the week
plt.figure(figsize=(8, 5))
df["Day_of_Week"].value_counts().sort_index().plot(kind='bar', color='cyan')
plt.title("Accidents Noticed Based on Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.show()

print(df["Accident_Severity"].unique())  # See all categories

#Objective 3:
# Define the color palette for each severity level
palette = {
    "Serious": "red",
    "Slight": "blue",
    "Fatal": "black",
    "Fetal": "purple"  # Only include if your data actually has "Fetal"
}

# Plotting accident locations
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="Accident_Severity", palette=palette)
plt.title("Accident Locations Based on Latitude and Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Accident Severity")
plt.show()

#Objective 4:
# Scatter plot to study correlation
plt.figure(figsize=(8, 6))
plt.scatter(df["Number_of_Vehicles"], df["Number_of_Casualties"], color='blue', alpha=0.5)
plt.title("Relationship Between Number of Casualties and Number of Vehicles")
plt.xlabel("Number of Vehicles Involved")
plt.ylabel("Number of Casualties")
plt.grid(True)
plt.show()


#Objective 5:
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Carriageway_Hazards", hue="Accident_Severity", palette="Set2")
plt.title("Impact of Carriageway Hazards on Accident Severity")
plt.xlabel("Carriageway Hazards")
plt.ylabel("Accident Frequency")
plt.xticks(rotation=45)
plt.legend(title="Severity")
plt.show()

#Objective 6:
# Filter data for only 'Dry' and 'Wet/Damp' road conditions
df_filtered = df[df['Road_Surface_Conditions'].isin(['Dry', 'Wet/Damp'])]

# Create two separate samples
dry_casualties = df_filtered[df_filtered['Road_Surface_Conditions'] == 'Dry']['Number_of_Casualties']
wet_casualties = df_filtered[df_filtered['Road_Surface_Conditions'] == 'Wet/Damp']['Number_of_Casualties']

# Remove any missing or null values
dry_casualties = dry_casualties.dropna()
wet_casualties = wet_casualties.dropna()

# Perform two-sample t-test (independent samples)
t_stat, p_value = ttest_ind(dry_casualties, wet_casualties, equal_var=False)  # Welch's t-test

# Display results
print("Two-Sample t-Test (Dry vs Wet Roads):")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in mean casualties between Dry and Wet roads.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("Conclusion: No significant difference in mean casualties between Dry and Wet roads.")

#Objective 7:
    # Count number of accidents in Urban and Rural areas
urban_count = df[df['Urban_or_Rural_Area'] == 'Urban'].shape[0]
rural_count = df[df['Urban_or_Rural_Area'] == 'Rural'].shape[0]

# Total number of accidents
total_count = urban_count + rural_count

# Number of accidents in each group
successes = [urban_count, rural_count]

# Total observations in each group
nobs = [total_count, total_count]

# Perform two-proportion z-test
z_stat, p_value = proportions_ztest(count=successes, nobs=nobs)

# Display results
print("Proportion Z-Test (Urban vs Rural Accidents):")
print(f"z-statistic = {z_stat:.4f}")
print(f"p-value = {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print("Conclusion: There is a significant difference in accident proportions between Urban and Rural areas.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("Conclusion: No significant difference in accident proportions between Urban and Rural areas.")

#Objective 8:
    
# Convert 'Accident Date' column to datetime format
df['Accident_Date'] = pd.to_datetime(df['Accident Date'], dayfirst=True)

# Create a new column for month or week
df['Month'] = df['Accident_Date'].dt.to_period('M')  # For monthly trend
# df['Week'] = df['Accident_Date'].dt.to_period('W')  # Uncomment for weekly trend

# Group by Month and Accident_Severity
monthly_trend = df.groupby(['Month', 'Accident_Severity']).size().unstack().fillna(0)

# Plotting
plt.figure(figsize=(12, 6))
for severity in monthly_trend.columns:
    plt.plot(monthly_trend.index.astype(str), monthly_trend[severity], label=severity)

plt.title("Monthly Accident Frequency by Severity Level")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.legend(title='Accident Severity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
