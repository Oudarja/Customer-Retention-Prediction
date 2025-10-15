def add_features(df):
    # +1 for current year to avoid division by zero
    df['avg_spend_per_year'] = df['Total_Spend'] / (df['Years_as_Customer'] + 1)
    return df