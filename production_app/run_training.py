from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import select_features
from src.sampling import apply_smote
from src.train import train_model, save_model
from sklearn.model_selection import train_test_split

df = load_data("/home/prince-raj/Documents/credit-card-fraud-detection/data/raw/Credit-Card.csv")
df = preprocess(df)

X, y = select_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_res, y_res = apply_smote(X_train, y_train)

model = train_model(X_res, y_res)
save_model(model)

