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

def anticlassification_gender(model, y_preds, df_test):
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

def group_fairness_gender(model, df_test):
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

def get_tpr_common(c_matrix):

    tn, fp, fn, tp = c_matrix.ravel()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    group_fairness_metric = (fp + tp) / (tn + fp + fn + tp)

    return {"tpr":tpr, "tnr":tnr, "fpr":fpr, "fnr":fnr, "P(Y = 1)":group_fairness_metric}

def train_separation_gender(df_train):
    df_train["Sex_male"] = np.random.binomial(1, 0.5, len(df_train))
    X_train = df_train.drop("Risk_bad", axis = 1).values
    y_train = df_train["Risk_bad"].values
    model = train_model(X_train, y_train, "sex_separation")
    return model

def test_group_fairness_gender(model, df_test, threshold):
    df_male = df_test[df_test["Sex_male"]==1.0]
    df_female = df_test[df_test["Sex_male"]==0.0]

    x_test_male = df_male[df_male.columns[:-1]]
    y_true_male = df_male[df_male.columns[-1]]

    x_test_female = df_female[df_female.columns[:-1]]
    y_true_female = df_female[df_female.columns[-1]]

    y_proba_male = model.predict_proba(x_test_male.values)[:,1]
    y_proba_female = model.predict_proba(x_test_female.values)[:,1]

    y_preds_male = np.where(y_proba_male >= threshold, 1, 0)
    y_preds_female = np.where(y_proba_female >= threshold, 1, 0)

    confusion_matrix_male = confusion_matrix(y_true=y_true_male, y_pred=y_preds_male)
    confusion_matrix_female = confusion_matrix(y_true=y_true_female, y_pred=y_preds_female)

    x_test = df_test[df_test.columns[:-1]]
    y_test = df_test[df_test.columns[-1]]
    y_proba = model.predict_proba(x_test.values)[:, 1]
    y_preds = np.where(y_proba >= threshold, 1, 0)
    acc = np.where(y_preds==y_test, 1, 0).sum() / len(y_test)

    print(threshold, acc)

    return confusion_matrix_male, confusion_matrix_female, acc

def get_age_mapping():
    age_mapping = {'student': [0, 0, 0],
        'young': [1, 0, 0],
        'adult': [0, 1, 0],
        'senior': [0, 0, 1]
        }
    age_cols = ["Age_cat_Young", "Age_cat_Adult", "Age_cat_Senior"]
    return age_mapping, age_cols

def create_dfs(df_test, age_cols, age_mapping):
    dfs = {}
    for k, v in age_mapping.items():

        young_condition = df_test[age_cols[0]] == v[0]
        adult_condition = df_test[age_cols[1]] == v[1]
        senior_condition = df_test[age_cols[2]] == v[2]

        dfs[k] = df_test.loc[(young_condition) & (adult_condition) & (senior_condition), :]

    return dfs


def get_anticlassification_train_dfs_age(df_train, df_test, age_cols, age_mapping):
    df_train_anticlassification = df_train.copy()

    df_train_anticlassification[age_cols] = age_mapping['student']
    display(df_train_anticlassification.head())

    df_test_anticlassification = df_test.copy()
    return df_train_anticlassification, df_test_anticlassification


def group_fairness_age(dfs, loaded_model):
    results = {}
    for key, df in dfs.items():
        print()
        print(key)
        X_test, y_test = get_X_and_y(df)
        y_pred = loaded_model.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        print("c_matrix\n", c_matrix)
        results[key] = get_tpr_common(c_matrix)
        
    return results

def get_group_fairness_age(dfs, og_model, file_suffix):
    group_fairness_dict = group_fairness_age(dfs, og_model)

    import json


    with open(f"group_fairness_age{file_suffix}.json", "w") as f:
        json.dump(group_fairness_dict, f, indent = 2)