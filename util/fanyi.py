import json
import random
import urllib.request
import http.client
import hashlib


def md5(str):  # 生成md5
    m1 = hashlib.new('md5')
    m1.update(str.encode('utf-8'))
    sign = m1.hexdigest()
    return sign


def en_to_zh(src):  # 英译中
    ApiKey = "20190426000291777"
    pwd = "buDANJCPliyt91vu2Enq"
    salt = str(random.randint(32768, 65536))
    all = ApiKey + src + salt + pwd
    sign = md5(all)
    src = src.replace(' ', '+')  # 生成sign前不能替换
    url = "http://api.fanyi.baidu.com/api/trans/vip/translate?q=" \
          + src + "&from=en&to=zh&appid=" + ApiKey + \
          "&salt=" + salt + "&sign=" + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', url)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result = response.read()

        data = json.loads(result)
        wordMean = data['trans_result'][0]['dst']
        return wordMean
    except Exception as e:
        print(e)
        return "出错了"


def zh_to_en(src):  # 中译英
    ApiKey = "20190426000291777"
    pwd = "buDANJCPliyt91vu2Enq"
    salt = str(random.randint(32768, 65536))
    all = ApiKey + src + salt + pwd
    sign = md5(all)
    src = src.replace(' ', '+')  # 生成sign前不能替换
    url = "http://api.fanyi.baidu.com/api/trans/vip/translate?q=" \
          + src + "&from=zh&to=en&appid=" + ApiKey + \
          "&salt=" + salt + "&sign=" + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', url)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result = response.read()

        data = json.loads(result)
        wordMean = data['trans_result'][0]['dst']
        return wordMean
    except Exception as e:
        print(e)
        return "出错了"


def main():
    choice = input("英语到汉语: 请输入1\n"
                   "中文到英文: 请输入2 \n"
                   "输入:")
    if choice == "1":
        while True:
            word = input("输入你想要搜索的单词:")
            print("translate......")
            target = en_to_zh(word)
            print(target)
    else:
        while True:
            word = input("输入你想要搜索的单词:")
            print("翻译......")
            target = zh_to_en(word)
            print(target)


if __name__ == '__main__':
    main()
