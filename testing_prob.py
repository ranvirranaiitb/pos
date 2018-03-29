import numpy as np
import math

def exp_fn(avg_latency,x):
	return 1-math.exp(-1/avg_latency)

tempsum = 0
N = 1000

for i in range(N):
	incomplete = True
	temp =0
	while incomplete:

		if(np.random.uniform(0,1)<exp_fn(100,temp)):
			tempsum += temp
			incomplete = False
		print(temp)
		temp += 1
	print(temp)

print(tempsum/N)
