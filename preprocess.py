import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the big CIC-ToN-IoT dataset
print("Loading dataset...")
df = pd.read_csv('CIC-ToN-IoT.csv')

# 2. We skip chronological sorting and go straight to sampling
# Let's take a 20% random subset of the total data to keep the size manageable,
# but we STRATIFY it so the 20% has the exact same Benign/Attack ratio as the whole CSV.
print("Taking a representative 20% subset...")
df_subset, _ = train_test_split(df, train_size=0.20, stratify=df['Label'], random_state=42)

# 3. Split the subset into Train (80%) and Test (20%)
# Again, stratifying ensures attacks are distributed equally to both sets.
print("Splitting into Train and Test...")
train_data, test_data = train_test_split(df_subset, test_size=0.20, stratify=df_subset['Label'], random_state=42)

# 4. Verify the distributions
print("\n--- TRAIN SET LABEL DISTRIBUTION ---")
print(train_data['Label'].value_counts())

print("\n--- TEST SET LABEL DISTRIBUTION ---")
print(test_data['Label'].value_counts())

# 5. Save the data
print(f"\nSaving train_data.csv ({len(train_data)} rows)...")
train_data.to_csv('train_data.csv', index=False)

print(f"Saving test_data.csv ({len(test_data)} rows)...")
test_data.to_csv('test_data.csv', index=False)

print("Done!")