import sklearn
import shap
import reader

X_train, X_valid, y_train, y_valid = reader.census()

knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)

f = lambda x: knn.predict_proba(x)[:,1]

med = X_train.median().values.reshape((1,X_train.shape[1]))

explainer = shap.KernelExplainer(f, med)

shap_values_single = explainer.shap_values(X.iloc[0,:], nsamples=1000)

shap_values = explainer.shap_values(X_valid.iloc[0:1000,:], nsamples=1000)

print(shap_values.shape)
print(explainer.expected_value)