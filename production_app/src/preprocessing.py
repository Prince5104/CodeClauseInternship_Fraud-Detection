from sklearn.preprocessing import StandardScaler

def preprocess(df):
    scaler = StandardScaler()
    df["normAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["normTime"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Amount", "Time"], axis=1, inplace=True)
    return df