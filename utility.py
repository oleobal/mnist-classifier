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
def getNicePercent(numb,decimals=0):
	return getNiceRound(numb,decimals)+"%"

def getNiceRound(numb,dec=0):
	numb = round(numb,dec)
	if dec==0:
		numb = int(numb)
	
	numb = str(numb)
	
	numb=numb+" "*(4-len(numb))
	return numb