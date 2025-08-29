import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
#Load Datasets
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [
        ['Bread', 'Butter'],
        ['Milk', 'Diaper', 'Juice', 'Cookies'],
        ['Butter', 'Diaper', 'Juice', 'Soda'],
        ['Bread', 'Milk', 'Diaper', 'Juice'],
        ['Bread', 'Butter', 'Diaper', 'Soda']
    ]
}
df = pd.DataFrame(data)
print("Prajwol Lab-2")
print("Initial Data:\n", df)
# Step 2: Convert dataset into a format suitable for the Apriori algorithm# Convert the list of items into one-hot encoded format
df_items = df['Items'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
print("\nOne-Hot Encoded Data:\n", df_items)
# Step 3: Apply the Apriori algorithm to find frequent itemsets# Use a minimum support threshold of 0.6 (at least 60% of transactions)
frequent_itemsets = apriori(df_items, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)
# Step 4: Generate association rules from the frequent itemsets# Use a minimum confidence threshold of 0.7 (at least 70% confidence)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nAssociation Rules:\n", rules)
# Step 5: Interpret the results# Display the rules in a simple format
for _, row in rules.iterrows():
  print(f"\nRule: {set(row['antecedents'])} -> {set(row['consequents'])}")
  print(f"Support: {row['support']:.2f}")
  print(f"Confidence: {row['confidence']:.2f}")
  print(f"Lift: {row['lift']:.2f}")