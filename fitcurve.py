import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

#====================== Selecione o modelo através dos valores de type (entre 1 e 6) =====================================================
type = 6
#=========================================================================================================================================
model_1 = r'MR = $k_{\mathrm{1}}$$\sigma_{\mathrm{3}}^{\mathrm{k_{\mathrm{2}}}}$'
model_2 = r'MR = $k_{\mathrm{1}}$$\theta^{\mathrm{k_{\mathrm{2}}}}$'
model_3 = r'MR = ($k_{\mathrm{1}}$ + $k_{\mathrm{2}}$ ($k_{\mathrm{3}}$ - $\sigma_{\mathrm{d}}^{\mathrm{k_{\mathrm{2}}}}$))$\sigma_{\mathrm{3}}^{\mathrm{k_{\mathrm{3}}}}$'
model_4 = r'MR = $k_{\mathrm{1}}$$\sigma_{\mathrm{3}}^{\mathrm{k_{\mathrm{2}}}}$$\sigma_{\mathrm{d}}^{\mathrm{k_{\mathrm{3}}}}$'
model_5 = r'MR = $k_{\mathrm{1}}$$\sigma_{\mathrm{3}}^{\mathrm{k_{\mathrm{2}}}}$$\sigma_{\mathrm{1}}^{\mathrm{k_{\mathrm{3}}}}$'
model_6 = r'MR = $k_{\mathrm{1}}$$\theta^{\mathrm{k_{\mathrm{2}}}}$$\sigma_{\mathrm{d}}^{\mathrm{k_{\mathrm{3}}}}$'

models = [model_1, model_2, model_3, model_4, model_5, model_6]

legendas = [[r'$\sigma_{\mathrm{3}}$ (MPa)', r'$\sigma_{\mathrm{d}}$ (MPa)'],
			[r'$\theta$ (MPa)', r'$\sigma_{\mathrm{d}}$ (MPa)'],
			[r'$\sigma_{\mathrm{3}}$ (MPa)', r'$\sigma_{\mathrm{d}}$ (MPa)'],
			[r'$\sigma_{\mathrm{3}}$ (MPa)', r'$\sigma_{\mathrm{d}}$ (MPa)'],
			[r'$\sigma_{\mathrm{3}}$ (MPa)', r'$\sigma_{\mathrm{1}}$ (MPa)'],
			[r'$\theta$ (MPa)', r'$\sigma_{\mathrm{d}}$ (MPa)']
			]

def model(sigmas, k1, k2, k3):
	sigma3, sigmad = sigmas
	if type == 1:
		k3 = 0
		return k1 * (sigma3**k2)
	if type == 2:
		teta = 2*sigma3 + sigma3 + sigmad
		k3 = 0
		return k1*(teta**k2)
	if type == 3:
		return (k1 + k2*(k3-(sigmad**k2)))*(sigma3**k3)
	if type == 4:
		return k1 * (sigma3**k2) * (sigmad**k3)
	if type == 5:
		teta = 2*sigma3 + sigma3 + sigmad
		sigma = sigma3 + sigmad
		return k1 * (sigma3**k2) * (sigma**k3)
	if type == 6:
		teta = 2*sigma3 + sigma3 + sigmad
		return k1 * (teta**k2) * (sigmad**k3)

#============ Para facilitar a escolha do arquivo, coloque o script numa pasta e crie outra pasta para os arquivos .txt ==================
caminho = os.getcwd() + '/Argiloso/MR_teste_Medio.txt'
#====================== Os dados devem ser colunas separadas por tab na ordem: σ3	σd	MR. Todos em MPa =================================
dataframe = pd.read_csv(caminho, header=None, sep='\t')
#=========================================================================================================================================

sigma3_data = dataframe[0].to_numpy()
sigmad_data = dataframe[1].to_numpy()
mr_data = dataframe[2].to_numpy()

# Transformação logarítmica dos dados
chute_inicial = [1, 1, 1]

print(f'Chutes iniciais: k1 = {chute_inicial[0]}, k2 = {chute_inicial[1]}, k3 = {chute_inicial[2]}')

# Dados combinados em uma tupla para curve_fit
sigmas3ed_data = np.array([sigma3_data, sigmad_data])

# Ajuste dos parâmetros usando curve_fit com os chutes iniciais
params, covariance = curve_fit(model, sigmas3ed_data, mr_data, p0=chute_inicial)

# Extraindo os parâmetros ajustados
k1, k2, k3 = params
print(f'Parâmetros ajustados: k1 = {k1}, k2 = {k2}, k3 = {k3}')

# Previsão dos valores de z usando os parâmetros ajustados
z_pred = model(sigmas3ed_data, k1, k2, k3)
# Cálculo do R²
ss_res = np.sum((mr_data - z_pred) ** 2)
ss_tot = np.sum((mr_data - np.mean(mr_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f'R² = {r_squared}')

# Gerando uma grade de pontos para x e y
x_range = np.linspace(min(sigma3_data), max(sigma3_data), 50)
y_range = np.linspace(min(sigmad_data), max(sigmad_data), 50)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Calculando os valores de z para cada ponto da grade usando os parâmetros ajustados
z_grid = model((x_grid, y_grid), k1, k2, k3)

#normalizando cores
norm = plt.Normalize(z_grid.min(), z_grid.max())
colors = plt.cm.Blues(norm(z_grid))

# Visualização da superfície ajustada e dos pontos de dados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sigma3_data, sigmad_data, mr_data, color='red', label="MR Experimental")
# Aplicando o colormap `Blues` aos valores de z
surf = ax.plot_surface(x_grid, y_grid, z_grid,
						facecolors=colors,
						alpha=0.6, rstride=1, cstride=1, linewidth=0, antialiased=False,
						label=rf'Modelo: {models[type-1]}'
						'\n'
						rf'R²={round(r_squared, 3)}'
						'\n'
						rf'k$_{{\mathrm{{1}}}}$={round(k1, 3)}, k$_{{\mathrm{{2}}}}$={round(k2, 3)}, k$_{{\mathrm{{3}}}}$={round(k3, 3)}')

# Adicionando a barra de cores
mappable = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
mappable.set_array(z_grid)
plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, alpha=0.6, label='MR (MPa)')

# Invertendo o eixo y
ax.invert_yaxis()

# Rotação e elevação
ax.view_init(elev=30, azim=-45)

ax.set_xlabel(legendas[type-1][0])
ax.set_ylabel(legendas[type-1][1])
ax.set_zlabel('MR (MPa)')
plt.legend()
plt.show()