#importamos nuestras blibliotecas
import numpy as np
import torch

# Definimos la red neuronal
class RNND(torch.nn.Module):
    def __init__(self, n_entradas, n_neuronas, n_salidas):
        super().__init__()
        self.capa_entrada = torch.nn.Linear(n_entradas, n_neuronas)
        self.capa_oculta = torch.nn.Linear(n_neuronas, n_neuronas)
        self.capa_salida = torch.nn.Linear(n_neuronas, n_salidas)

#funcion que nos dice como van propagandose los datos atraves de la red
    def forward(self, x):

        x = self.capa_entrada(x)
        h = x
        lista_salidas = []
        for t in range(x.shape[0]):
            h = self.capa_oculta(h)
            salida_t = self.capa_salida(h)
            lista_salidas.append(salida_t)

        return torch.stack(lista_salidas)

# Generamos los datos de entrenamiento
x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
y_train = torch.sin(x_train)

# Inicializamos la red neuronal
model = RNND(1, 10, 1)

# Entrenamos la red neuronal
optimizar = torch.optim.Adam(model.parameters())
for epoch in range(100):
    y_pred = model(x_train)
    perdida = torch.mean((y_pred - y_train) ** 2)
    optimizar.zero_grad()
    perdida.backward()
    optimizar.step()

# Evaluamos la red neuronal
x_test = torch.tensor(np.random.rand(10, 1), dtype=torch.float32)
y_test = torch.sin(x_test)
y_pred = model(x_test)

# Imprimimos las predicciones de manera estructurada
for i in range(y_pred.shape[0]):
    print(f'Muestra {i + 1}:')
    for j in range(y_pred.shape[1]):
        print(f'  Predicción {j + 1}: {y_pred[i, j, 0].item()}')
    print()
