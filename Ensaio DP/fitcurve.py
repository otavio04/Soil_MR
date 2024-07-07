import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

def converter_formato_dados(dados_original):

	colunas = []
	tensao_inicial = 40
	sigma3 = []
	sigmad = []
	n = []
	dp = []

	for i in range(len(dados_original[0, :])):
		colunas.append(dados_original[:, i])
	
	for i in range(len(dados_original[0, :])-1):
		for j, k in enumerate(colunas[i+1]):#	i é a coluna (indo da segunda a 7), j é a linha (indo da primeira até a 29)
			if k != -1:#							AQUI OS DADOS NÃO PODEM CONTER O VALOR -1, ESTES REPRESENTAM VAZIOS NO ENSAIO
				dp.append(k)#				Adicionando os dados de DP
				n.append(colunas[0][j])#		Adicionando os dados de ciclo

				if i%2 == 0:
					sigma3.append(tensao_inicial)
					sigmad.append(tensao_inicial)
				else:
					sigma3.append(tensao_inicial)
					sigmad.append(tensao_inicial*2)

		if i%2 != 0:
			tensao_inicial = tensao_inicial + 40
	
	return np.array(sigma3), np.array(sigmad), np.array(n), np.array(dp)

def model(variables, psi1, psi2, psi3, psi4):
	sigma3, sigmad, n = variables
	return psi1 * (sigma3**psi2) * (sigmad**psi3) * (n**psi4)

#============ Para facilitar a escolha do arquivo, coloque o script numa pasta e crie outra pasta para os arquivos .txt ==================
caminho = os.getcwd() + '/Argiloso/DP_testes_original.txt'
#====================== Os dados devem ser colunas separadas por tab na ordem: σ3	σd	MR. Todos em MPa =================================
dados = np.loadtxt(caminho, delimiter='\t', skiprows=1)
#=========================================================================================================================================
s3_data, sd_data, n_data, dp_data = converter_formato_dados(dados)

# Dados Iniciais para fazer o ajuste
chute_inicial = [1, 1, 1, 1]

# print(f'Chutes iniciais: psi1 = {chute_inicial[0]}, psi2 = {chute_inicial[1]}, psi3 = {chute_inicial[2]}, psi4 = {chute_inicial[3]}')

# Dados combinados em uma tupla para curve_fit
variables_data = np.array([s3_data, sd_data, n_data])

# Ajuste dos parâmetros usando curve_fit com os chutes iniciais
params, covariance = curve_fit(model, variables_data, dp_data, p0=chute_inicial)

casas_decimais = 3
# Extraindo os parâmetros ajustados
psi1, psi2, psi3, psi4 = params
# print(f'Parâmetros ajustados: psi1 = {round(psi1, casas_decimais)}, psi2 = {round(psi2, casas_decimais)}, psi3 = {round(psi3, casas_decimais)}, psi4 = {round(psi4, casas_decimais)}')

# Previsão dos valores de z usando os parâmetros ajustados
z_pred = model(variables_data, psi1, psi2, psi3, psi4)
# Cálculo do R²
ss_res = np.sum((dp_data - z_pred) ** 2)
ss_tot = np.sum((dp_data - np.mean(dp_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# print(f'R² = {round(r_squared, casas_decimais)}')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

legenda_ciclos = ['40 x 40 kPa', '40 x 80 kPa', '80 x 80 kPa', '80 x 160 kPa', '120 x 120 kPa', '120 x 240 kPa']
cores_ciclos = ['red', 'orange', 'green', 'blue', 'purple', 'black']

for i in range(len(dados[0, :]) - 1):
	if i == 3:
		ax[0].scatter(dados[0:24, 0], dados[0:24, i+1], color=cores_ciclos[i], alpha=0.4, label=legenda_ciclos[i])
	else:
		ax[0].scatter(dados[:, 0], dados[:, i+1], color=cores_ciclos[i], alpha=0.4, label=legenda_ciclos[i])

ax[0].set_title("Ciclo de Carga vs DP Experimental")
ax[0].set_xlabel("Número de Ciclos (N)")
# ax[0].set_xlabel(r"$Log_{\mathrm{10}}$(Ciclo)")
ax[0].set_ylabel("DP Experimental (cm)")
ax[0].legend(title=r'Pares de tensão ($\sigma_{\mathrm{3}}$ vs $\sigma_{\mathrm{d}}$)')
# ax[0].set_xscale('log')


model_1 = r'DP = $\psi_{\mathrm{1}}$$\sigma_{\mathrm{3}}^{\mathrm{\psi_{\mathrm{2}}}}$$\sigma_{\mathrm{d}}^{\mathrm{\psi_{\mathrm{3}}}}$$N^{\mathrm{\psi_{\mathrm{4}}}}$'

#Linha y = x
x_line = np.linspace(min(dp_data), max(dp_data), 100)
y_line = x_line 

ax[1].scatter(dp_data, z_pred, color='blue', alpha=0.4, label=f"Teste vs Modelo ({model_1})\n"r"$\psi_{\mathrm{1}}$"f"={round(psi1, 2)}, "r"$\psi_{\mathrm{2}}$"f"={round(psi2, 2)}, "r"$\psi_{\mathrm{3}}$"f"={round(psi3, 2)}, "r"$\psi_{\mathrm{4}}$"f"={round(psi4, 2)}")
ax[1].plot(x_line, y_line, color='blue', label=f"y = x\nR²: {round(r_squared, 2)}")
ax[1].set_title("DP Experimental vs Calculado")
ax[1].set_xlabel("DP Experimental (cm)")
ax[1].set_ylabel("DP Calculada (cm)")
ax[1].legend()
ax[1].set_aspect('equal')
ax[1].set_ylim(-0.01, max(max(z_pred), max(dp_data))*1.05)
ax[1].set_xlim(-0.01, max(max(z_pred), max(dp_data))*1.05)
ax[1].grid(True, color='#888888', linestyle='-', linewidth=0.1)

#Deixar as duas subplots com o mesmo tamanho
ax[1].set_box_aspect(1)
ax[0].set_box_aspect(1)

plt.show()
