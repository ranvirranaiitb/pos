

def var_majority(len,str):
	alpha = 0.6
	if str=='increasing':
		temp = 1 - alpha*(0.5**len)
	elif str =='decreasing':
		temp = 0.5 + alpha*(0.5**len)
	return temp