
from random import Random

NUM_SAMPLES_TO_GENERATE = 300

fileHandle = open('randomRGBSamples.csv', 'w')

rand = Random()
rand.seed(12345)

for i in range(NUM_SAMPLES_TO_GENERATE):
	colorIndex = rand.randint(1,500) % 3
	r = 0
	g = 0
	b = 0
	if (colorIndex == 0): # Red
		r = rand.randint(230, 255)
		g = rand.randint(0, 30)
		b = rand.randint(0, 30)
	elif (colorIndex == 1): # Green 
		r = rand.randint(0, 30)
		g = rand.randint(230, 255)
		b = rand.randint(0, 30)
	else: # Blue
		r = rand.randint(0, 30)
		g = rand.randint(0, 30)
		b = rand.randint(230, 255)
	
	rgbStr = " ".join(map(lambda p: str(p), [r, g, b]))
	fileHandle.write(str(colorIndex) + "," + rgbStr + '\n') 

fileHandle.close()
