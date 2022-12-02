import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_gauss = pd.read_csv('../hyperparam_data/data_gauss.csv')
df_rbf = pd.read_csv('../hyperparam_data/data_rbf.csv')
df_laplace = pd.read_csv('../hyperparam_data/data_laplace.csv')

idx = np.argmax(df_gauss["gauss_preds_list"])
max_pred = df_gauss["gauss_preds_list"][idx]
argmax_sigma = df_gauss["sigma_gauss_list"][idx]
fig,ax = plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(10)
ax.scatter([argmax_sigma],[max_pred],color="red")
ax.annotate(f"Maxima : Acc = {round(max_pred,3)} | Sigma = {argmax_sigma}",(argmax_sigma,max_pred))
plt.plot(df_gauss["sigma_gauss_list"], df_gauss["gauss_preds_list"])
plt.xlabel("Sigma_Gauss")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy vs Sigma Hyperparam for Gaussian Kernel")
plt.savefig("../graphs/AccVsSigmaGauss.png")

idx1 = np.argmax(df_rbf["rbf_preds_list"])
max_pred1 = df_rbf["rbf_preds_list"][idx1]
argmax_gamma1 = df_rbf["gamma_rbf_list"][idx1]
fig,ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
ax.scatter([argmax_gamma1],[max_pred1],color="red")
ax.annotate(f"Maxima : Acc = {round(max_pred1,3)} | Gamma = {argmax_gamma1}",(argmax_gamma1,max_pred1))
plt.plot(df_rbf["gamma_rbf_list"], df_rbf["rbf_preds_list"])
plt.xlabel("Gamma_RBF")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy vs Gamma Hyperparam for RBF Kernel")
plt.savefig("../graphs/AccVsGammaRBF.png")

idx2 = np.argmax(df_laplace["laplace_preds_list"])
max_pred2 = df_laplace["laplace_preds_list"][idx2]
argmax_sigma2 = df_laplace["sigma_laplace_list"][idx2]
fig,ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
ax.scatter([argmax_sigma2],[max_pred2],color="red")
ax.annotate(f"Maxima : Acc = {round(max_pred2,3)} | Sigma = {argmax_sigma2}",(argmax_sigma2,max_pred2))
plt.plot(df_laplace["sigma_laplace_list"], df_laplace["laplace_preds_list"])
plt.xlabel("Sigma_Laplace")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy vs Sigma Hyperparam for Laplace Kernel")
plt.savefig("../graphs/AccVsSigmaLaplace.png")