import xgboost


class ModelXGBRegression:
    def __init__(self, seed=1, n_jobs=-1, n_estimators=500, subsample=1, colsample_bytree=1, max_depth=5,
                 tree_method='hist'):
        self.model = xgboost.XGBRegressor(n_jobs=n_jobs,
                                          max_depth=max_depth,
                                          subsample=subsample,
                                          colsample_bytree=colsample_bytree,
                                          n_estimators=n_estimators,
                                          seed=seed,
                                          tree_method=tree_method)

    def fit_model(self, x_train, y_train, model_name=None, path=None):
        self.model.fit(X=x_train, y=y_train)
        if model_name:
            self.model.save_model(path + model_name)

    def fit_model_increment(self, x_train, y_train, model_name, path):
        self.model.fit(X=x_train, y=y_train, xgb_model=path+model_name)
        self.model.save_model(path+model_name)

    def predict(self, x_to_predict):
        return self.model.predict(x_to_predict)
