# Program to merge NASA battery degradation CSV files into one combined dataset
# Author: Kenneth Kamogelo Baloyi
# Date 03 July 2025
# Merge.py

#import necessary libraries
import pandas as pd
import os

#list of names of battery files to merge
battery_ids = ['B0005', 'B0006', 'B0007', 'B0018', 'B0025',
                'B0027', 'B0028', 'B0030', 'B0031', 'B0032'] 
all_dataframes = []

#Load and tag each battery dataset
for battery in battery_ids:
    filename = f"{battery}.csv" #variable for filename
    if os.path.exists(filename): #check if specified file exists in the cwd
        df = pd.read_csv(filename) #read file
        df['battery'] = battery #column that holds battery ID
        all_dataframes.append(df) #add columns in the dataframe
        print(f" Loaded {filename} ({len(df)} rows)")
    else:
        print(f" File not found: {filename} — skipping.")

#Merge all data into one DataFrame
merged_df = pd.concat(all_dataframes, ignore_index=True)

#Drop 'datetime' column if present
merged_df = merged_df.drop(columns='datetime', errors='ignore')

#remove duplicates
merged_df = merged_df.drop_duplicates()

#remove missing values
merged_df = merged_df.dropna()

# Remove invalid or extreme values 
merged_df = merged_df[merged_df['capacity'] > 0]
merged_df = merged_df[(merged_df['voltage_measured'] >= 2.5) & (merged_df['voltage_measured'] <= 4.3)]
merged_df = merged_df[(merged_df['current_measured'] >= -2) & (merged_df['current_measured'] <= 2)]
merged_df = merged_df[(merged_df['temperature_measured'] >= 10) & (merged_df['temperature_measured'] <= 45)]

#  Add cumulative time per battery
merged_df = merged_df.sort_values(by=['battery', 'cycle', 'time'])  # Ensure order is correct
merged_df['time_cumulative'] = merged_df.groupby('battery')['time'].cumsum()

#Final stats
print(" Cleaned DataFrame shape:", merged_df.shape)
print(merged_df.describe())

#Save cleaned data
merged_df.to_csv("Test_cleaned_discharge_data.csv", index=False)
print(" Cleaned discharge data saved to 'cleaned_discharge_data.csv'")
