import http.client
import hashlib
import urllib.request
import random
import json

appid = '20190426000291777'  # 你的appid
secretKey = 'buDANJCPliyt91vu2Enq'  # 你的密钥

httpClient = None
# myurl = '/api/trans/vip/translate'
myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
# 输入的单词
q = 'apple'

# 输入英文输出中文
fromLang = 'en'
toLang = 'zh'
salt = random.randint(32768, 65536)
sign = appid + q + str(salt) + secretKey
m1 = hashlib.new('md5')
m1.update(sign.encode('utf-8'))
sign = m1.hexdigest()
# m1 = hashlib.new('md5',sign).hexdigest()
# m1.update(sign)
# sign = m1.hexdigest()
myurl = myurl + '?q=' + urllib.request.quote(
    q) + '&from=' + fromLang + '&to=' + toLang + '&appid=' + appid + '&salt=' + str(salt) + '&sign=' + sign
try:
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)
    # response是HTTPResponse对象
    response = httpClient.getresponse()
    result = response.read()

    data = json.loads(result)
    wordMean = data['trans_result'][0]['dst']
    print(wordMean)

except Exception as e:
    print(e)
finally:
    if httpClient:
        httpClient.close()