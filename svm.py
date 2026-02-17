#Aqui está todas as bibliotecas necessárias para rodar o código.
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Aqui está a biblioteca que será utilizada para visualizar o SVM.
base_dados_iris = datasets.load_iris()

#Aqui temos uma função simples que constrói o gráfico e os pontos distribuídos no plano.
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


#Aqui nós estamos separando as variáveis que serão usadas para fazer a previsão.
X = base_dados_iris.data[:, :2]

#Aqui nós temos a variável alvo (Target), isto é, a variável que o modelo deve prever.
Y = base_dados_iris.target

#Aqui estamos definindo que os valores alvo não podem iguais a 2 (Iris-virginica). Isso é importante, pois, se não definirmos esse parâmetro, você estará tentando construir um plano 2D com dados em 3D.
mask = (Y != 2)

X = X[mask]
Y = Y[mask]

#Aqui estamos definindo os dados. Sobre esses dados, estamos separando 20% deles para testes. Além disso, estamos definindo a randomização dos dados em 42, de modo a evitar discrepâncias extremans conforme o código é rodado várias vezes.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Aqui estamos definindo o tipo de algoritmo usado e o tipo do Kernel, além do coeficiente C. Lembre-se, se o coeficiente for muito alto, caímos em Underfitting. Se for muito baixo, caímos em Overfitting.
modelo_svm = SVC(kernel='linear', C=1.0)

#Aqui estamos treinando o modelo com os dados de treino.
modelo_svm.fit(X_train, Y_train)

#Aqui estamos prevendo os valores de teste a partir do modelo que nós treinamos.
Y_pred = modelo_svm.predict(X_test)

#Por final, estamos construindo o gráfico que exibe a forma como os dados se comportam de acordo com os vetores. Note que os dados se separam por uma "linha invisível" muito nítida.
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='autumn')
plot_svc_decision_function(modelo_svm)
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('SVM: Hiperplano e Vetores de Suporte (Círculos Pretos)')
plt.show()


