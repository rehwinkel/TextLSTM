from keras.models import model_from_yaml
import numpy as np

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

mymodel = model_from_yaml(open("model.json", "r").read())
mymodel.load_weights("model.h5")

inval = np.array([textToVec("once upon a time")])
for i in range(500):
	inval = np.append(inval, mymodel.predict(inval))
	inval = inval.reshape(1, -1, len(letters))

print(vecToText(inval.reshape(-1, len(letters))))