import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

base_dados_iris = datasets.load_iris()

def plot_svc_decision_function(model, ax=None):

    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')


X = base_dados_iris.data[:, :2]
Y = base_dados_iris.target

mask = (Y != 2)

X = X[mask]
Y = Y[mask]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

modelo_svm = SVC(kernel='linear', C=1.0)
modelo_svm.fit(X_train, Y_train)

Y_pred = modelo_svm.predict(X_test)
print(classification_report(Y_test, Y_pred))


plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
plot_svc_decision_function(modelo_svm)
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('SVM: Hiperplano e Vetores de Suporte (Círculos Pretos)')
plt.show()


