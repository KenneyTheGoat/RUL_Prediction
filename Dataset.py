#Dataset.py
#Program to convert NASA battery degradation MATLAB files into CSV files compatible with excel 
# Author: Kenneth Kamogelo Baloyi
# Date: 03 July 2025
# Dataset.py

#Import necessary libraries
import datetime
import pandas as pd
from scipy.io import loadmat #package that handles .mat files
import os

print("Input which battery you need to extract data from. Choose from the following")
print("Batteries: B0005, B0006, B0007...B0056")

B = input("Enter battery ID: ")


#Function that reads in a .mat file and converts it into a .csv
def disch_data(battery):
    file_path = battery + '.mat'  #assuming files are in the same working directory
    if not os.path.exists(file_path): #check if file is present in the cwd
        raise FileNotFoundError(f"MAT file '{file_path}' not found in current directory.")
    
    mat = loadmat(file_path)
    print(f"Loaded {file_path}, extracting discharge cycles...")

    disdataset = []
    capacity_data = []
    c = 0

    for i in range(len(mat[battery][0, 0]['cycle'][0])): #Loop through the top level matrix 
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge': #Extract only discharge data points because they are ones that contain battery capacity measurements
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                          int(row['time'][0][1]),
                                          int(row['time'][0][2]),
                                          int(row['time'][0][3]),
                                          int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            
            for j in range(len(data[0][0]['Voltage_measured'][0])): #2nd level matrix
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]

                disdataset.append([c + 1, ambient_temperature, date_time, capacity,
                                   voltage_measured, current_measured,
                                   temperature_measured, current_load,
                                   voltage_load, time]) #add discharge fields into the dataframe
                
                capacity_data.append([c + 1, ambient_temperature, date_time, capacity]) #add capacity fields from only the discharge phase and not the entire capacity column
            c += 1
    
    print(f"Extracted {c} discharge cycles.")
    return pd.DataFrame(data=disdataset,
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                 'capacity', 'voltage_measured', 'current_measured',
                                 'temperature_measured', 'current', 'voltage', 'time']) #return dataframe with specified headers

#Run extraction
discharge_df = disch_data(B)

#Save to CSV
output_file = B + ".csv"
discharge_df.to_csv(output_file, index=False)
print(f"Discharge data saved to: {output_file}")
