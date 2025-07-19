import pandas as pd

# Load the original cleaned dataset
df = pd.read_csv("Test_cleaned_discharge_data.csv")

# Get the minimum count among all battery types
min_count = df["battery"].value_counts().min()

# Downsample each class and concatenate
balanced_df = (
    df.groupby("battery")
      .apply(lambda group: group.sample(n=min_count, random_state=42))
      .reset_index(drop=True)
)


# Save the new balanced and shuffled dataset
balanced_df.to_csv("Test_balanced_discharge_data.csv", index=False)
print(f"Balanced and shuffled dataset saved with {min_count} samples per battery type.")
