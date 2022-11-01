# %%
import pandas as pd
from plot import plot
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning, ValueWarning
import os
import time
from multiprocessing import Pool
import statsforecast.arima as sfar
from statsforecast.arima import Arima, AutoARIMA, arima_string, auto_arima_f, predict_arima, forecast_arima, fitted_arima
# %%

# 1 dia
window_size = 96
# Multiplicador pra não deixar o desvio muito pequeno
# No cusum os limites recomendados são 4 desvios, com a folga de meio desvio dá mais ou menos isso
s = 3.2
min_std = 0
max_std = 15
# Valor inicial caso tenha anomalia desde o começo
std = min_std

# %%

def timed_arima(ix):
	i, x = ix
	try:
		model.fit(y=y)
		pred = model.predict(1)['mean'][0]
	except Exception as e:
		pred = x[-1]
	if (i % 1000) == 0:
		print('Tempo de execução de 1000 itens', time.time())
	return pred

# %%
def async_arima(iy):
	i, y = iy
	model = AutoARIMA()
	try:
		model.fit(y=y)
		pred = model.predict(1)['mean'][0]
	except Exception as e:
		pred = y[-1]
	if i % 100 == 0:
		print(i)
	return i, pred

def async_arima(x):
	try:
		pred = ARIMA(x, order=(1,0,0)).fit().forecast()[0]
	except Exception as e:
		pred = x[-1]
	return pred

def sep_anomalies(anomalies):
	anom_start = -1
	isolated_anomalies = []
	continuous_anomalies = []
	for i, anomaly in enumerate(anomalies):
		# Se o próximo for anomalia
		if i < (len(anomalies) - 1) and anomalies[i+1] == anomaly+1:
			# E o anterior não for ou for o primeiro da lista
			if anomalies[i-1] != anomaly-1 or i == 0:
				# É começo de uma contínua
				anom_start = anomaly
		# Se o anterior é anomalia e o próximo não for ou for o fim da lista, é fim de contínuo
		elif i > 0 and anomalies[i-1] == anomaly-1:
			continuous_anomalies.append((anom_start, anomaly))
			anom_start = -1
		# Se o próximo e o anterior não forem anomalia, é isolado
		else:
			isolated_anomalies.append(anomaly)
	return isolated_anomalies, continuous_anomalies

def seq_data(data, window_size=96):
	data_X = []
	data_Y = []
	for i in range(len(data) - (len(data) % window_size) - window_size):
		data_X.append(data[i:i+window_size])
		try:
			data_Y.append(data[i+window_size])
		except:
			(i+window_size)
	return np.array(data_X), np.array(data_Y)

def clear_warns():
	simplefilter("ignore", category=ConvergenceWarning)
	simplefilter("ignore", category=HessianInversionWarning)
	simplefilter("ignore", category=RuntimeWarning)
	simplefilter("ignore", category=ValueWarning)
	simplefilter("ignore", category=UserWarning)

def mp_arima_numba(data: np.array) -> (list, AutoARIMA):
	if len(data) == 0:
		return None
	model = AutoARIMA(period=window_size, max_p=2, max_q=2, max_d=2, stepwise=False, parallel=True, approximation=True, trace=True)
	model.fit(data)
	return list(model.model_.model['arma'])[:3], model
 
if __name__ == '__main__':
	# %%

	# %%
	clear_warns()

	# %%

	df = pd.read_csv(f'../../Dados/por estacao/26425235/export_automaticas_26425235_pressao.csv')

	# df = df.loc[df['date'].apply(lambda d: d.split('-')[0] == '2016')]
		
	data = np.array(df['pressao'].to_list()[:5000])

	# Tirando NaN e inf e etc...
	# Substitui pelo último valor
	data = np.array([data[i] if np.isfinite(data[i]) else data[i-1] for i in range(len(data))])

	# Dados separado em janelas
	reshaped_data, targets = seq_data(data, window_size=window_size)
	# processes = 8
	# chunksize, extra = divmod(len(data), processes * 4)
	# if extra:
	# 	chunksize += 1

	standart_d = [max(min(np.std(x) * s, max_std), min_std) for x in reshaped_data]

	print('Prevendo série')
	prev = fitted_arima(Arima(data, order=(1, 0, 0)))
	print(len(prev))
	print(prev)
	# t = time.perf_counter()
	# predicted_series = model.predict_in_sample(1)['mean'].to_list()[window_size:]
	# print(f'fim: {time.perf_counter() - t}')
	# print('Achando anomalias')
	# anom_positions = [i for (i, prediction), std in zip(enumerate(predicted_series), standart_d) if abs(prediction - targets[i]) > std]

	# series = data[window_size:]

	# isolated, continuous = sep_anomalies(anom_positions)

	# try:
	# 	path = os.path.join(f'./results/data/pressao')
	# 	os.makedirs(path)
	# except Exception as e:
	# 	pass
	
	# try:
	# 	path = os.path.join(f'./results/images/pressao')
	# 	os.makedirs(path)
	# except Exception as e:
	# 	pass

	# plot(
	# 	series,
	# 	detected_anomalies=isolated,
	# 	continuous_anomalies=continuous,
	# 	sec_plots=[predicted_series],
	# 	std_dev=standart_d,
	# 	save=True,
	# 	img_name=f'./results/images/pressao/{station}_pressao_numba.jpg',
	# 	show=False
	# )

	# result = pd.DataFrame({
	# 	'real_value': [data[i + window_size] for i in range(len(predicted_series))],
	# 	'predictions': predicted_series,
	# 	'anomaly': [1 if i in anom_positions else 0 for i in range(len(predicted_series))],
	# 	'std': standart_d,
	# })
	# result.to_csv(
	# 	f'./results/data/pressao/{station}_pressao_numba.csv',
	# 	sep=';',
	# 	index=False
	# )

	# # print('Método antigo')
	# # t = time.perf_counter()
	# # with Pool() as pool:
	# # 	predicted_series = list(pool.imap_unordered(async_arima, enumerate(reshaped_data), chunksize))
	# # 	predicted_series.sort(key=lambda ix: ix[0])
	# # 	predicted_series = [ix[1] for ix in predicted_series]

	# # print('fim', time.perf_counter() - t)
	# # anom_positions = [i for (i, prediction), std in zip(enumerate(predicted_series), standart_d) if abs(prediction - targets[i]) > std]

	# # series = data[window_size:]

	# # isolated, continuous = sep_anomalies(anom_positions)

	# # try:
	# # 	path = os.path.join('./results/data/{}'.format('pressao'))
	# # 	os.makedirs(path)
	# # except Exception as e:
	# # 	pass
	
	# # try:
	# # 	path = os.path.join('./results/images/{}'.format('pressao'))
	# # 	os.makedirs(path)
	# # except Exception as e:
	# # 	pass

	# # plot(series, detected_anomalies=isolated, continuous_anomalies=continuous, sec_plots=[predicted_series], std_dev=standart_d, save=True, img_name='./results/images/{}/{}_{}_old.jpg'.format('pressao', station, 'pressao'), show=False)

	# # result = pd.DataFrame({
	# # 	'real_value': [data[i + window_size] for i in range(len(predicted_series))],
	# # 	'predictions': predicted_series,
	# # 	'anomaly': [1 if i in anom_positions else 0 for i in range(len(predicted_series))],
	# # 	'std': standart_d,
	# # })
	# # result.to_csv('./results/data/{}/{}_{}_old.csv'.format('pressao', station, 'pressao'), sep=';',index=False)