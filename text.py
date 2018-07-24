from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import numpy as np
import time

letters = "abcdefghijklmnopqrstuvwxyz ,.!?'"

def textToVec(text):
	global letters

	data = []
	for t in text:
		if t.lower() in letters:
			v = letters.index(t.lower())
			vec = np.zeros(len(letters))
			vec[v] = 1
			data.append(vec)
		else:
			print(t)
	return np.array(data)

def vecToText(vect):
	out = ""
	global letters
	for fv in vect:
		maxv = 0
		maxi = 0
		for v in range(len(fv)):
			if fv[v] > maxv:
				maxv = fv[v]
				maxi = v
		out += letters[maxi]
	return out

def loadTextData(filename, sql):
	global letters

	if sql < 1:
		raise TypeError("Sequence length must be at least 1")

	text = open(filename, "r").read()
	text = text.replace("\n", " ")

	data = textToVec(text)

	sequences = []
	sqi = sql + 1
	for i in range(len(data) - sqi):
		sequence = data[i:i+sqi]
		sequences.append(list(sequence))
	sequences = np.array(sequences)
	sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], len(letters))

	x_data = sequences[:, :-1]
	y_data = sequences[:, -1]

	return (x_data, y_data)

x_data, y_data = loadTextData("words.txt", 50)
print(x_data.shape)
print(y_data.shape)

model = Sequential()
model.add(LSTM(input_shape=(None, len(letters)), units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=len(letters)))
model.add(Activation("linear"))

start = time.time()
model.compile(loss="mse", optimizer="rmsprop")
print("Compilation time is {}".format(time.time()-start))

model.fit(x_data, y_data, epochs=100, batch_size=30)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

inval = np.array([textToVec("ge")])
for i in range(50):
	inval = np.append(inval, model.predict(inval))
	inval = inval.reshape(1, -1, len(letters))

print(vecToText(inval.reshape(-1, len(letters))))