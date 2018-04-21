import requests
import pandas as pd 
from pandas import DataFrame, Series
import codecs
import json

def getAbbr(filepath):
	df = pd.read_csv(filepath, names=['abbr'])
	return df 

def writeJson(path, abbr, offset, content):
	name = path+abbr+"_"+str(offset)+".json"
	f = codecs.open(name, 'w','utf8')
	f.write(content)
	f.close()

def get_response(url, headers):
	res = requests.get(url, headers=headers)
	return res 

def getURL(url_pre, expr, offset):
	url = url_pre+"expr="+expr+"&count=1000&offset="+offset+"&attributes=Id,Ti,Y,CC,AA.AuN,AA.AuId,AA.AfId,AA.AfN,AA.S,F.FN,F.FId,J.JN,J.JId,C.CN,C.CId,RId,FP,LP"
	return url

if __name__ == '__main__':
	path = "./result/"
	#url = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=And(Composite(C.CN=='aaai'),Y=2016)&count=1000&offset=1000&attributes=Ti,Y,CC,AA.AuN,AA.AuId,AA.AfId,AA.AfN,AA.S"

	headers = {
		'Ocp-Apim-Subscription-Key': '52c7ed2264ce412b9f09dc20529e1ffc',
	}
	df = pd.read_csv("input.csv", names=['abbr'])
	for i in df.abbr:
		dic = {}
		abbr = i.lower()
		dic['expr'] = "And(Composite(C.CN=='" + abbr +"'),Y>2004)"
		#dic['count'] = str(1000)
		dic['offset'] = 0
		#dic['attributes'] = 'Ti,Y,CC,AA.AuN,AA.AuId,AA.AfId,AA.AfN,AA.S,F.FN,F.FId,J.JN,J.JId,C.CN,C.CId,RId,FP,LP'
		url_pre = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?"
		while True:
			url = getURL(url_pre, dic['expr'], str(dic['offset']))
			#print url
			res = get_response(url=url, headers=headers)
			print "Status: ", res
			try: 
				data = json.loads(res.text)
				length = len(data['entities'])
				if length > 0:
					writeJson(path, abbr, dic['offset'], res.text)
					
				if length < 1000:
					#if length<1000, it proves no more next page's papers
					print "Conference: "+abbr +" Fininshed! The count is: ", str(dic['offset']+length)
					print "-------------------------------------------------------------------------------------"
					break
				dic['offset'] += 1000
			except:
				print "Conference: "+abbr +" Failed!!"
				print "-----------------------------------------------------------------------"
				continue