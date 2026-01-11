def select_features(df):
    x = df.drop("Class", axis=1)
    y = df["Class"]
    return x, y