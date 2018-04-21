#encoding=utf-8
import requests
import codecs
import json

#https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=Composite(AA.AuN=='jaime teevan')&count=2&attributes=Ti,Y,CC,AA.AuN,AA.AuId

#url = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=Composite(AA.AuN=='jaime teevan')&count=20&attributes=Ti,Y,CC,AA.AuN,AA.AuId,C.CN"

headers = {
	'Ocp-Apim-Subscription-Key': '52c7ed2264ce412b9f09dc20529e1ffc',
}

url = "https://westus.api.cognitive.microsoft.com/academic/v1.0/evaluate?expr=And(Composite(C.CN=='aaai'),Y=2016)&count=10&offset=0&attributes=Ti,Y,CC,AA.AuN,AA.AuId,AA.AfId,AA.AfN,AA.S,F.FN,F.FId,J.JN,J.JId,C.CN,C.CId,RId"

res = requests.get(url, headers=headers)
print "返回码：",res
#print "内容："
#print res.text

f = codecs.open('res.json', 'w', 'utf8')
f.write(res.text)
f.close()
d = json.loads(res.text)

#Load
# ff = codecs.open('res.json', 'r', 'utf8')
# data = json.load(ff)
# ff.close()