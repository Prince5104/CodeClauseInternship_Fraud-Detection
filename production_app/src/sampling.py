from imblearn.over_sampling import SMOTE

def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)