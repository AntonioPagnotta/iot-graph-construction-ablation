import pandas as pd

# 1. Load the CSV
df = pd.read_csv('outputs/edge_labels_5ep.csv')

# 2. Drop the column (replace 'Column_To_Drop' with your actual column name)
# axis=1 tells pandas to drop a column, not a row
df = df.drop('probability', axis=1)

df = df.drop('edge_type', axis=1)
# 3. Save as a new file
# index=False ensures pandas doesn't write row numbers into your new file
df.to_csv('output.csv', index=False)