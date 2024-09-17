import pandas as pd
import os


'''
WORLD BANK DATA
'''

file_path = os.path.join(os.getcwd(), 'world.csv')
df = pd.read_csv(file_path, encoding='latin-1')

#removes double spacing
def clean_column_names(df):

  cleaned_names = df.columns.to_series().apply(lambda x: x.strip().replace('  ', ' '))
  df = df.rename(columns=cleaned_names)
  return df

# Clean column names
df = clean_column_names(df)

# Access and print cleaned column names
print("Cleaned column names:", list(df.columns))

# Save the cleaned DataFrame to the same CSV file
df.to_csv("world_cleaned.csv", index=False)

######################################################################################################################

'''
GLOBAL PARTICIPATION FILE
'''

file_path = os.path.join(os.getcwd(), 'global_participation.csv')
cleaned_file_path = os.path.join(os.getcwd(), 'global_participation_no_duplicates.csv')

df = pd.read_csv(file_path, encoding='latin-1')

# Find duplicates
duplicates = df[df.duplicated()]

# Print duplicates if any
if not duplicates.empty:
    print("Duplicates found in the dataset:")
    print(duplicates)
else:
    print("No duplicates found in the dataset.")

# Remove duplicates
df_cleaned = df.drop_duplicates()

# Save the cleaned data to a new file
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"\nRemoved duplicates. The cleaned data has been saved to {cleaned_file_path}.")

# Display the number of rows before and after cleaning
print(f"\nNumber of rows before cleaning: {len(df)}")
print(f"Number of rows after cleaning: {len(df_cleaned)}")

######################################################################################################################

'''
HISTORICAL OPERATORS-OWNERS ASSIGNMENT
''' 

current_directory = os.getcwd()
rystad_path = os.path.join(current_directory, 'global_merged_draft.csv')
hist_path = os.path.join(current_directory, 'global_participation_no_duplicates.csv')

rystad_df = pd.read_csv(rystad_path, encoding='latin-1')
hist_rystad_df = pd.read_csv(hist_path, encoding='latin-1')

year = 2022 #works for any selected year in the dataset

def assign_historical_operator(hist_df, rystad_df, hist_col_name=year, operator_col_name='Historical Operator'):
   
    # Merge on ID to link historical data to Rystad data
    merged_df = rystad_df.merge(hist_df[['RE ID', operator_col_name]], how='left', on='RE ID')

    # Group by ID and find the row with the maximum value in 'hist_col_name'
    max_value_df = merged_df.groupby('RE ID')[operator_col_name].max().reset_index()

    # Efficient vectorized assignment using merge
    rystad_df = rystad_df.merge(max_value_df[['RE ID', operator_col_name]], how='left', on='RE ID')
    rystad_df.rename(columns={operator_col_name: f'Historical Operator {year}'}, inplace=True)
    
    return rystad_df

def add_owner_details(rystad_df, hist_df):
    
    rystad_df[f'All Hist Owners {year}'] = rystad_df['RE ID'].apply(lambda x: 
        ', '.join(hist_df[(hist_df['RE ID'] == x) & (hist_df[f'{year}'] > 0)]['Historical Company'].tolist()) if len(hist_df[(hist_df['RE ID'] == x) & (hist_df[f'{year}'] > 0)]) > 0 else ''
    )
    return rystad_df

# Example usage
rystad_df_with_histop = assign_historical_operator(hist_rystad_df, rystad_df)
rystad_df_with_histown = add_owner_details(rystad_df_with_histop, hist_rystad_df)

# Save the updated DataFrame to a new CSV file
output_path = os.path.join(current_directory, 'global_merged_draft_op.csv')
rystad_df_with_histown.to_csv(output_path, index=False)

print(f"Updated DataFrame saved to {output_path}")

######################################################################################################################


