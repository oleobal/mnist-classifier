"""
utility function for displaying a progress bar of <length> chars
"""
def progressBar(i, maxi, length):
	nbchars = round((i/maxi)*(length-5))
	result = "["+nbchars*"="+(length-5-nbchars)*"-"+"]"
	pcent = round((i/maxi)*100)
	if pcent<10:
		result+="0"+str(pcent)+"%"
	elif pcent<100:
		result+=str(pcent)+"%"
	else:
		result+=str(pcent)
	return result


"""
returns a justified string for the given number (1-100)
"""
def getNicePercent(numb):
	numb = str(round(numb,2))
	
	numb=numb+" "*(5-len(numb))+"%"
	return numb
