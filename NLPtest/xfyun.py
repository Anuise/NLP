import time
import urllib.request
import urllib.parse
import json
import hashlib
import base64



def main():
    context='小明和小美去旅館打砲'
    #BODY填需要分析的文本，長度限制為500(簡中)
    body = urllib.parse.urlencode({'text': context }).encode('utf-8')

    #API接口 http://ltpapi.xfyun.cn/v1/{func}  
    #{func}替換成想要的功能，如中文分詞(cws) URL則改為>>http://ltpapi.xfyun.cn/v1/cws
    #詳見>>https://doc.xfyun.cn/rest_api/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%9F%BA%E7%A1%80%E5%A4%84%E7%90%86.html
    url = 'http://ltpapi.xfyun.cn/v1/sdgp'
    
    #申請的KEY
    api_key = '7a21e7b6800f623322880f488b66369a'
    param = {"type": "dependent"}

    #申請的應用ID(APPID)，非API KEY
    x_appid = '5c495e5e'

    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    x_time = int(int(round(time.time() * 1000)) / 1000)
    x_checksum = hashlib.md5(api_key.encode('utf-8') + str(x_time).encode('utf-8') + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib.request.Request(url, body, x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    print(context)
    print(result.decode('utf-8'))
    return


if __name__ == '__main__':
    main()