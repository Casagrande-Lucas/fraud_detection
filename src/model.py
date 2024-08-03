from sklearn.ensemble import RandomForestClassifier


def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model
