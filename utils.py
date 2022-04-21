def train_model(X_train, y_train, model_name):
    param_test1 = {
    'max_depth':[3,5,6,10],
    'min_child_weight':[3,5,10],
    'gamma':[0.0, 0.1, 0.2, 0.3, 0.4],
    # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10],
    'subsample':[i/100.0 for i in range(75,90,5)],
    'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    }

    #Creating the classifier
    model_xg = XGBClassifier(random_state=2)

    grid_search = GridSearchCV(model_xg, param_grid=param_test1, cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    pkl.dump(best_model, open(f"{model_name}.pickle", "wb"))

    return best_model

def anticlassification_sex(model, y_preds, df_test):
    test_df = df_test.copy()
    test_df["Sex_male"] = test_df["Sex_male"].map({0.0:1.0, 1.0:0.0})
    y_preds_new = model.predict(test_df.values)
    change_count = np.where(y_preds == y_preds_new, 0, 1)
    col = pd.Series(change_count)
    print(col.describe())
    return change_count.sum() / len(y_preds)

def train_anticlassification(df_train, drop_feat):
    df_train[drop_feat] = [0 for _ in range(len(df_train))]
    print(df_train.head())
    # exit()
    X_train = df_train.drop("Risk_bad", axis = 1).values
    y_train = df_train["Risk_bad"].values
    model = train_model(X_train, y_train, "sex_anticlassification")
    return model

def group_fairness(model, df_test):
    df_male = df_test[df_test["Sex_male"]==1.0]
    df_female = df_test[df_test["Sex_male"]==0.0]

    x_test_male = df_male[df_male.columns[:-1]]
    y_true_male = df_male[df_male.columns[-1]]

    x_test_female = df_female[df_female.columns[:-1]]
    y_true_female = df_female[df_female.columns[-1]]

    y_preds_male = model.predict(x_test_male.values)
    y_preds_female = model.predict(x_test_female.values)

    confusion_matrix_male = confusion_matrix(y_true=y_true_male, y_pred=y_preds_male)
    confusion_matrix_female = confusion_matrix(y_true=y_true_female, y_pred=y_preds_female)

    return confusion_matrix_male, confusion_matrix_female

def get_tpr(c_matrix):

    tn, fp, fn, tp = c_matrix.ravel()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    group_fairness_metric = (fp + tp) / (tn + fp + fn + tp)

    return {"tpr":tpr, "tnr":tnr, "fpr":fpr, "fnr":fnr, "P(Y = 1)":group_fairness_metric}

def train_group_fairness(df_train):
    df_train.loc[df_train["Risk_bad"] == 1, "Sex_male"] = np.random.binomial(1, 0.5, len(df_train.loc[df_train["Risk_bad"] == 1, "Sex_male"]))
    X_train = df_train.drop("Risk_bad", axis = 1).values
    y_train = df_train["Risk_bad"].values
    model = train_model(X_train, y_train, "sex_group_fairness")
    return model

def train_separation(df_train):
    df_train.loc[df_train["Risk_bad"] == 0, "Sex_male"] = np.random.binomial(1, 0.5, len(df_train.loc[df_train["Risk_bad"] == 0, "Sex_male"]))
    X_train = df_train.drop("Risk_bad", axis = 1).values
    y_train = df_train["Risk_bad"].values
    model = train_model(X_train, y_train, "sex_separation")
    return model