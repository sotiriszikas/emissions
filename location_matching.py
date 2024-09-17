import pandas as pd
import os
from fuzzywuzzy import fuzz
from geopy.distance import geodesic

'''
ANALYSIS FOR ALL THE COUNTRY NAMES IN THE LIST
'''

year = 2022 #works for any selected year in the dataset 
countries = ["Russia", "Iraq", "Iran", "Algeria", "Venezuela", "United States", "Mexico", "Libya", "Nigeria"] #selected countries for analysis

# Function to check if field name matches any asset name with at least 45% similarity
# and the distance between their coordinates is less than 50 kilometers for the closest match
def is_field_in_assets(field_name, field_coords, asset_list, asset_coords, re_ids, threshold=45, max_distance_km=50):
    field_name = str(field_name).lower()
    closest_match = (None, float('inf'), None, 0)  # (asset, distance, re_id, fuzz_ratio)
    second_closest_match = (None, float('inf'), None, 0)  # (asset, distance, re_id, fuzz_ratio)

    for asset, asset_coord, re_id in zip(asset_list, asset_coords, re_ids):
        asset = str(asset).lower()
        fuzz_ratio = fuzz.ratio(field_name, asset)
        if fuzz_ratio >= threshold:
            distance = geodesic(field_coords, asset_coord).kilometers
            distance = round(distance, 2)  # Round the distance to 2 decimal places
            if distance < max_distance_km:
                if distance < closest_match[1]:
                    second_closest_match = closest_match
                    closest_match = (asset, distance, re_id, fuzz_ratio)
                elif distance < second_closest_match[1]:
                    second_closest_match = (asset, distance, re_id, fuzz_ratio)
            elif closest_match[0] is None or distance < closest_match[1]:
                second_closest_match = closest_match
                closest_match = (asset, distance, re_id, fuzz_ratio)

    second_distance_message = second_closest_match[1] if second_closest_match[1] < max_distance_km else f'>{max_distance_km}km'

    if closest_match[0] is not None:
        return ('yes', closest_match[0], closest_match[2], closest_match[1], second_closest_match[0], second_closest_match[2], second_distance_message, closest_match[3], second_closest_match[3])
    return ('no', 'no asset', '', None, 'no asset', '', None, 0, 0)

current_directory = os.getcwd()

file_path_1 = os.path.join(current_directory, 'GGFR_cleaned.csv')
file_path_2 = os.path.join(current_directory, 'global_merged_draft_op.csv')

data_1 = pd.read_csv(file_path_1)
data_2 = pd.read_csv(file_path_2, encoding='ISO-8859-1')


directory_path = os.path.join(current_directory, f'countries_field_names_match_{year}')
os.makedirs(directory_path, exist_ok=True)

for country in countries:
    data_1_filtered = data_1[(data_1['COUNTRY'] == country) & (data_1['Year'] == year) & (data_1['Field Type'] != 'LNG')].copy()
    data_2_filtered = data_2[(data_2['Asset Type'] == 'Field') & (data_2['Country'] == country)]

    if data_1_filtered.empty or data_2_filtered.empty:
        print(f"No matching data found for {country} in {year}. Skipping...")
        continue

    asset_list = list(data_2_filtered['Asset'])
    asset_coords = list(zip(data_2_filtered['Latitude'], data_2_filtered['Longitude']))
    re_ids = list(data_2_filtered['RE ID'])

    data_1_filtered['result'] = data_1_filtered.apply(
        lambda row: is_field_in_assets(row['Field Name'], (row['Latitude'], row['Longitude']), asset_list, asset_coords, re_ids), axis=1
    )
    
    max_distance_km = 50  
    data_1_filtered['Criteria met (1st)'] = data_1_filtered['result'].apply(lambda x: 'yes' if x[0] == 'yes' and x[3] < max_distance_km else 'no')
    data_1_filtered['Criteria met (2nd)'] = data_1_filtered['result'].apply(lambda x: 'yes' if x[4] != 'no asset' and x[6] != f'>{max_distance_km}km' and x[6] < max_distance_km else 'no')
    data_1_filtered['1st asset'] = data_1_filtered['result'].apply(lambda x: x[1])
    data_1_filtered['1st fuzz_ratio'] = data_1_filtered['result'].apply(lambda x: x[7])
    data_1_filtered['1st RE ID'] = data_1_filtered['result'].apply(lambda x: x[2])
    data_1_filtered['1st distance_km'] = data_1_filtered['result'].apply(lambda x: x[3])
    data_1_filtered['2nd asset'] = data_1_filtered['result'].apply(lambda x: x[4])
    data_1_filtered['2nd fuzz_ratio'] = data_1_filtered['result'].apply(lambda x: x[8])
    data_1_filtered['2nd RE ID'] = data_1_filtered['result'].apply(lambda x: x[5])
    data_1_filtered['2nd distance_km'] = data_1_filtered['result'].apply(lambda x: x[6])
    data_1_filtered.drop('result', axis=1, inplace=True)  # Remove the intermediate 'result' column

    # Reorder columns
    columns_order = ['Field Name', 'Criteria met (1st)', 'Criteria met (2nd)', '1st asset', '1st fuzz_ratio', '1st RE ID', '1st distance_km', 
                     '2nd asset', '2nd fuzz_ratio', '2nd RE ID', '2nd distance_km'] + \
                    [col for col in data_1_filtered.columns if col not in ['Field Name', 'Criteria met (1st)', 'Criteria met (2nd)', '1st asset', '1st fuzz_ratio', '1st RE ID', '1st distance_km', '2nd asset', '2nd fuzz_ratio', '2nd RE ID', '2nd distance_km']]
    data_1_filtered = data_1_filtered[columns_order]

    output_path = os.path.join(directory_path, f'FNM_field_asset_{country}_{year}.csv')
    data_1_filtered.to_csv(output_path, index=False)
    print(f"File for {country} {year} was saved at {output_path}")

print(f"FNM process completed for all the countries.")

######################################################################################################################

'''
MERGING COUNTRIES' GENERATED FILES 
'''

directory_path = os.path.join(current_directory, f'countries_field_names_match_{year}')
output_merged_file = os.path.join(current_directory, f'FNM_merged_field_asset_{year}.csv')

# Initialize an empty DataFrame to hold the merged data
merged_df = pd.DataFrame()

# Iterate over all files in the directory and merge them
for country in countries:
    file_path = os.path.join(directory_path, f'FNM_field_asset_{country}_{year}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    else:
        print(f"File for {country} not found. Skipping...")

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_merged_file, index=False)
print(f"Merged file saved at {output_merged_file}")

######################################################################################################################


'''
MATCHING HISTORICAL DATA (OWNERS AND OPERATORS)
'''
merged_file_path = os.path.join(current_directory, f'FNM_merged_field_asset_{year}.csv')  # World Bank
ry_path = os.path.join(current_directory, 'global_merged_draft_op.csv')

merged_field_name_match_file = pd.read_csv(merged_file_path, encoding='latin-1')
ry_data = pd.read_csv(ry_path, encoding='ISO-8859-1')

historical_operator_dict = ry_data.set_index('RE ID')['Historical Operator 2022'].to_dict()
historical_owners_dict = ry_data.set_index('RE ID')['All Hist Owners 2022'].to_dict()
current_owners_dict = ry_data.set_index('RE ID')['Ownership'].to_dict()

# Map the historical operators and owners to the merged data for 1st RE ID
merged_field_name_match_file['1st Historical Operator 2022'] = merged_field_name_match_file['1st RE ID'].map(historical_operator_dict)
merged_field_name_match_file['1st Historical Owners 2022'] = merged_field_name_match_file['1st RE ID'].map(historical_owners_dict)
merged_field_name_match_file['1st Current Owners'] = merged_field_name_match_file['1st RE ID'].map(current_owners_dict)

# Map the historical operators and owners to the merged data for 2nd RE ID
merged_field_name_match_file['2nd Historical Operator 2022'] = merged_field_name_match_file['2nd RE ID'].map(historical_operator_dict)
merged_field_name_match_file['2nd Historical Owners 2022'] = merged_field_name_match_file['2nd RE ID'].map(historical_owners_dict)
merged_field_name_match_file['2nd Current Owners'] = merged_field_name_match_file['2nd RE ID'].map(current_owners_dict)

# Get the list of columns before adding the new ones
columns_before = list(merged_field_name_match_file.columns)

# Find the positions of the "1st distance_km" and "2nd distance_km" columns
position_1st = columns_before.index('1st distance_km') + 1
position_2nd = columns_before.index('2nd distance_km') + 1

# Create a list of new columns for 1st and 2nd RE ID
new_columns_1st = ['1st Historical Operator 2022', '1st Historical Owners 2022', '1st Current Owners']
new_columns_2nd = ['2nd Historical Operator 2022', '2nd Historical Owners 2022', '2nd Current Owners']

# Rearrange the columns
columns_after_1st = columns_before[:position_1st] + new_columns_1st + columns_before[position_1st:position_2nd] + new_columns_2nd + columns_before[position_2nd:]

# Remove duplicates if the new columns were added in positions already containing other new columns
columns_after_1st = pd.Index(columns_after_1st).unique()

# Reindex the DataFrame with the new column order
merged_field_name_match_file = merged_field_name_match_file.reindex(columns=columns_after_1st)

# Save the updated DataFrame to a new CSV file
output_file_path = os.path.join(current_directory, f'FNM_merged_field_asset_op_{year}.csv')
merged_field_name_match_file.to_csv(output_file_path, index=False, encoding='latin-1')

print(f"Updated file saved at {output_file_path}")

######################################################################################################################

'''
CALCULATING PERCENTILES AND DISTANCE RATIOS
FINAL CSV IS GENERATED
'''

filename = f'FNM_merged_field_asset_op_{year}.csv'
file_path = os.path.join(current_directory, filename)

df = pd.read_csv(file_path, encoding='latin-1')
df['bcm_percentile_rank'] = df['bcm'].rank(pct=True) * 100

# Assuming 'second_distance_km' and 'distance_km' might contain strings
df['2nd distance_km'] = pd.to_numeric(df['2nd distance_km'], errors='coerce')
df['1st distance_km'] = pd.to_numeric(df['1st distance_km'], errors='coerce')

# Calculate distance_ratio (handling potential NaN values)
df['distance_ratio'] = df['2nd distance_km'] / df['1st distance_km'].where(df['1st distance_km'] != 0, other=np.nan)

# Round 'bcm_percentile_rank' and 'distance_ratio' to 2 decimal places
df['bcm_percentile_rank'] = df['bcm_percentile_rank'].round(2)
df['distance_ratio'] = df['distance_ratio'].round(2)

def is_match(row, wb_operator_col, ry_ownership_col, ry_operator_col): 
    if pd.isna(row[wb_operator_col]) or pd.isna(row[ry_ownership_col]):
        return 'missing', 'missing'

    field_operator = str(row[wb_operator_col]) 
    asset_ownership = row[ry_ownership_col]
    asset_operator = row[ry_operator_col]
    
    # Use fuzzy matching with a threshold of 45
    if fuzz.ratio(field_operator, asset_operator) >= 45:
        return 'yes', 'yes'
    else:
        if field_operator in asset_ownership:
            return 'no', 'yes'
        else:
            return 'no', 'no'

# Apply the function to the 'Closest Asset Ownership' column
df[['cl.asset hop. match', 'cl.asset how. match']] = df.apply(lambda row: is_match(row, 'Field Operator', '1st Historical Owners 2022', '1st Historical Operator 2022'), axis=1, result_type='expand')

# Apply the function to the 'Second Closest Asset Ownership' column
df[['2nd cl.asset hop. match', '2nd cl.asset how. match']] = df.apply(lambda row: is_match(row, 'Field Operator', '2nd Historical Owners 2022', '2nd Historical Operator 2022'), axis=1, result_type='expand')

df['Field Operator'] = df['Field Operator'].fillna('Unknown')  # Replace with a default value
output_filename = f"FNM_final_{year}.csv"
output_file_path = os.path.join(current_directory, output_filename)

df.to_csv(output_file_path, index=False)
print(f"Comparison for all names in Asset Ownership column completed and results written to the file: {output_filename}")

######################################################################################################################