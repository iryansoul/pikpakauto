from quart import Quart, render_template, request, Response, stream_with_context, jsonify
from typing import Any, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from rich import print_json
import numpy as np
import asyncio
import aiohttp
import hashlib
import random
import json
import time
import re
import cv2


# 生成随机IP
def generate_random_ipv4():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))


# 生成随机xid
def uuid():
    return ''.join([random.choice('0123456789abcdef') for _ in range(32)])


# 滑块计算函数
def image_assemble(sum_rows, sum_cols, channels, part_num):
    final_matrix = np.zeros((sum_rows, sum_cols, channels), np.uint8)

    row_slices = [slice(0, 75), slice(75, 150), slice(150, 225), slice(225, 300)]
    col_slices = [slice(0, 150), slice(150, 300), slice(300, 450), slice(450, 600)]

    for row_index, row_slice in enumerate(row_slices):
        for col_index, col_slice in enumerate(col_slices):
            final_matrix[row_slice, col_slice] = part_num[row_index * 4 + col_index]

    return final_matrix


def getSize(p):
    sum_rows = p.shape[0]
    sum_cols = p.shape[1]
    channels = p.shape[2]
    return sum_rows, sum_cols, channels


async def get_img(deviceid, pid, traceid):
    url = f'https://user.mypikpak.com/pzzl/image?deviceid={deviceid}&pid={pid}&traceid={traceid}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as response:
            return await response.read()


def re_image_assemble(part, img):
    # 划分图像块
    parts = {
        '0,0': img[0:75, 0:150],
        '0,1': img[75:150, 0:150],
        '0,2': img[150:225, 0:150],
        '0,3': img[225:300, 0:150],
        '1,0': img[0:75, 150:300],
        '1,1': img[75:150, 150:300],
        '1,2': img[150:225, 150:300],
        '1,3': img[225:300, 150:300],
        '2,0': img[0:75, 300:450],
        '2,1': img[75:150, 300:450],
        '2,2': img[150:225, 300:450],
        '2,3': img[225:300, 300:450],
        '3,0': img[0:75, 450:600],
        '3,1': img[75:150, 450:600],
        '3,2': img[150:225, 450:600],
        '3,3': img[225:300, 450:600]
    }

    # 根据索引组装图像块
    part_nu = [parts[j] for j in part]

    return part_nu


def corp_image(img):
    img2 = img.sum(axis=2)
    (row, col) = img2.shape
    row_top, raw_down, col_top, col_down = 0, row - 1, 0, col - 1

    for r in range(row):
        if img2[r].sum() < 740 * col:
            row_top = r
            break
    for r in range(row - 1, 0, -1):
        if img2[r].sum() < 740 * col:
            raw_down = r
            break
    for c in range(col):
        if img2[:, c].sum() < 740 * row:
            col_top = c
            break
    for c in range(col - 1, 0, -1):
        if img2[:, c].sum() < 740 * row:
            col_down = c
            break

    new_img = img[row_top:raw_down + 1, col_top:col_down + 1]
    return new_img


def get_reimage(images):
    cropped_img = {}
    for i in range(16):
        re_num = corp_image(images[i])
        cropped_img[i] = cv2.resize(re_num, (150, 75))
    return cropped_img


def start_pass_verify(sum_rows, sum_cols, channels, img, result, i):
    part_1 = result["frames"][i]['matrix'][0]
    part_2 = result["frames"][i]['matrix'][1]
    part_3 = result["frames"][i]['matrix'][2]
    part_4 = result["frames"][i]['matrix'][3]
    part = []
    part.extend(part_1)
    part.extend(part_2)
    part.extend(part_3)
    part.extend(part_4)

    part_num = re_image_assemble(part, img)
    part_num = get_reimage(part_num)
    f = image_assemble(sum_rows, sum_cols, channels, part_num)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(edges)[0]
    return len(lines)


async def pass_verify(deviceid, pid, traceid, result):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=12)
    try:
        img_data = await get_img(deviceid, pid, traceid)
        img_re = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        sum_rows, sum_cols, channels = getSize(img_re)
    except Exception as e:
        print(e)
        return None

    data_list = list(range(12))
    start_pass_verify_re = partial(start_pass_verify, sum_rows, sum_cols, channels, img_re, result)
    tasks = [loop.run_in_executor(executor, start_pass_verify_re, data) for data in data_list]

    len_num = await asyncio.gather(*tasks)
    num, lines = min(enumerate(len_num), key=lambda x: x[1] if x[1] is not None else float('inf'))
    print(f'滑动次数: {num} 次')
    return num


# 滑块及加密函数
def r(e, t):
    n = t - 1
    if n < 0:
        n = 0
    r = e[n]
    u = r['row'] // 2 + 1
    c = r['column'] // 2 + 1
    f = r['matrix'][u][c]
    l = t + 1
    if l >= len(e):
        l = t
    d = e[l]
    p = l % d['row']
    h = l % d['column']
    g = d['matrix'][p][h]
    y = e[t]
    m = 3 % y['row']
    v = 7 % y['column']
    w = y['matrix'][m][v]
    b = i(f) + o(w)
    x = i(w) - o(f)
    return [s(a(i(f), o(f))), s(a(i(g), o(g))), s(a(i(w), o(w))), s(a(b, x))]


def i(e):
    return int(e.split(",")[0])


def o(e):
    return int(e.split(",")[1])


def a(e, t):
    return str(e) + "^⁣^" + str(t)


def s(e):
    t = 0
    n = len(e)
    for r in range(n):
        t = u(31 * t + ord(e[r]))
    return t


def u(e):
    t = -2147483648
    n = 2147483647
    if e > n:
        return t + (e - n) % (n - t + 1) - 1
    if e < t:
        return n - (t - e) % (n - t + 1) + 1
    return e


def c(e, t):
    return s(e + "⁣" + str(t))


def img_jj(e, t, n):
    return {
        'ca': r(e, t),
        'f': c(n, t)
    }


def compute_hash(input_str):
    # 左移位函数
    def rotate_left(value, shift):
        return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF

    # 16位溢出乘法函数
    def multiply_16_bits(a, b):
        high_a = (a >> 16) & 0xFFFF
        low_a = a & 0xFFFF
        high_b = (b >> 16) & 0xFFFF
        low_b = b & 0xFFFF
        return (((high_a * low_b + low_a * high_b) & 0xFFFF) << 16) + (low_a * low_b)

    # 用特定常量初始化哈希数组
    hash_array = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a]

    # 处理输入字符串
    for i in range(len(input_str)):
        hash_array[i % 4] = (hash_array[i % 4] ^ multiply_16_bits(ord(input_str[i]), 0x9e3779b9)) & 0xFFFFFFFF
        hash_array[i % 4] = rotate_left(hash_array[i % 4], 13)

    # 进一步混合哈希数组的值
    for j in range(8):
        for k in range(4):
            hash_array[k] = (hash_array[k] + multiply_16_bits(hash_array[(k + 1) % 4], 0x85ebca6b)) & 0xFFFFFFFF
            hash_array[k] = rotate_left(hash_array[k], 17)
            hash_array[k] = (hash_array[k] ^ hash_array[(k + 2) % 4]) & 0xFFFFFFFF

    # 最终组合哈希数组的值
    result = (hash_array[0] ^ hash_array[1] ^ hash_array[2] ^ hash_array[3]) & 0xFFFFFFFF
    result = multiply_16_bits(result, 0xc2b2ae35)
    result = (result ^ (result >> 16)) & 0xFFFFFFFF

    # 将结果转换为十六进制字符串，并在必要时填充前导零
    return format(result, '08x')


def md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()


def get_sign(xid, t):
    e = [
        {"alg": "md5", "salt": "KHBJ07an7ROXDoK7Db"},
        {"alg": "md5", "salt": "G6n399rSWkl7WcQmw5rpQInurc1DkLmLJqE"},
        {"alg": "md5", "salt": "JZD1A3M4x+jBFN62hkr7VDhkkZxb9g3rWqRZqFAAb"},
        {"alg": "md5", "salt": "fQnw/AmSlbbI91Ik15gpddGgyU7U"},
        {"alg": "md5", "salt": "/Dv9JdPYSj3sHiWjouR95NTQff"},
        {"alg": "md5", "salt": "yGx2zuTjbWENZqecNI+edrQgqmZKP"},
        {"alg": "md5", "salt": "ljrbSzdHLwbqcRn"},
        {"alg": "md5", "salt": "lSHAsqCkGDGxQqqwrVu"},
        {"alg": "md5", "salt": "TsWXI81fD1"},
        {"alg": "md5", "salt": "vk7hBjawK/rOSrSWajtbMk95nfgf3"}
    ]
    md5_hash = f"YvtoWO6GNHiuCl7xundefinedmypikpak.com{xid}{t}"
    for item in e:
        md5_hash += item["salt"]
        md5_hash = md5(md5_hash)
    return md5_hash


# PIKPAK运行逻辑
class PIKPAK:
    def __init__(self, incode: str) -> None:
        self.incode = incode
        self.IP = generate_random_ipv4()
        self.xid = uuid()
        self.t = str(int(time.time()))
        self.sign = get_sign(self.xid, self.t)

    # 临时邮箱函数
    async def get_mail(self):
        characters = 'abcdefghijklmnopqrstuvwxyz0123456789'
        demo = ''.join(random.choice(characters) for _ in range(10))
        body = {
            'name': demo,
            'domain': 'chmg9999.top',
        }
        url = f"https://chmg9999.top/api/new_address"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, ssl=False) as response:
                self.response_data = await response.json()
                self.mail = self.response_data['address']
                self.jwt = self.response_data['jwt']
                print('获取临时邮箱信息:')
                print_json(json.dumps(self.response_data, indent=4))
                return self.response_data

    async def get_code(self):
        url = 'https://chmg9999.top/api/mails?limit=20&offset=0'
        headers = {
            'Authorization': f'Bearer {self.jwt}'
        }
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers, ssl=False) as response:
                    response_data = await response.json()
                    if not response_data['results']:
                        print('等待接收邮件...')
                        await asyncio.sleep(1)  # 等待1秒钟
                        continue
                    else:
                        print('接收到邮件!!!')
                        regex = re.compile(r'<h2>(\d{6})</h2>')  # 正则表达式
                        match = regex.search(response_data['results'][0]['raw'])  # 匹配6位数字的正则表达式
                        if match:
                            print('提取到验证码:', match.group(1))
                            self.code = match.group(1)
                            break  # 匹配到的6位数字
                        else:
                            raise Exception('未提取到验证码!!!')
            return self.code  # 如果没有找到，返回异常捕获

    # 网络请求函数
    async def init(self):
        url = 'https://user.mypikpak.com/v1/shield/captcha/init'
        body = {
            "client_id": "YvtoWO6GNHiuCl7x",
            "action": "POST:/v1/auth/verification",
            "device_id": self.xid,
            "captcha_token": "",
            "meta": {
                "email": self.mail
            }
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'accept-language': 'zh-CN',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                if 'url' in response_data:
                    print('初始安全验证:')
                    print_json(json.dumps(response_data, indent=4))
                    self.captcha_token = response_data['captcha_token']
                else:
                    print('IP或者邮箱频繁,请更换IP或者稍后重试...')
                    raise Exception(response_data.get('error_description', 'Unknown error'))
                return response_data

    async def get_image(self):
        url = "https://user.mypikpak.com/pzzl/gen"
        params = {
            "deviceid": self.xid,
            "traceid": ""
        }
        headers = {
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, ssl=False) as response:
                self.data = await response.json()
                self.frames = self.data['frames']
                self.traceid = self.data['traceid']
                self.pid = self.data['pid']
        return self.data

    async def get_verify(self):
        async with aiohttp.ClientSession() as session:
            num = await pass_verify(self.xid, self.pid, self.traceid, self.data)
            json_data = img_jj(self.frames, int(num), self.pid)
            f = json_data['f']
            d = compute_hash(self.pid + self.xid + str(f))
            npac = json_data['ca']
            params = {
                'pid': self.pid,
                'deviceid': self.xid,
                'traceid': self.traceid,
                'f': f,
                'n': npac[0],
                'p': npac[1],
                'a': npac[2],
                'c': npac[3],
                'd': d,
            }
            headers = {
                'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, '
                              'like Gecko)'
                              'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36'
            }
            async with session.get(f"https://user.mypikpak.com/pzzl/verify", params=params, headers=headers,
                                   ssl=False) as response:
                response_data = await response.json()
                return response_data

    async def get_new_token(self):
        headers = {
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f'https://user.mypikpak.com/credit/v1/report?deviceid={self.xid}&captcha_token={self.captcha_token}&type=pzzlSlider'
                    f'&result=0&data={self.pid}&traceid={self.traceid}', headers=headers, ssl=False) as response:
                responseData = await response.json()
                self.captcha_token = responseData['captcha_token']
        return responseData

    async def verification(self):
        url = 'https://user.mypikpak.com/v1/auth/verification'
        body = {
            "email": self.mail,
            "target": "ANY",
            "usage": "REGISTER",
            "locale": "zh-CN",
            "client_id": "YvtoWO6GNHiuCl7x"
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'accept-language': 'zh-CN',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-captcha-token': self.captcha_token,
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('发送验证码:')
                print_json(json.dumps(response_data, indent=4))
                self.verification_id = response_data['verification_id']
                return response_data

    async def verify(self):
        url = 'https://user.mypikpak.com/v1/auth/verification/verify'
        body = {
            "verification_id": self.verification_id,
            "verification_code": self.code,
            "client_id": "YvtoWO6GNHiuCl7x"
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'accept-language': 'zh-CN',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('验证码验证结果:')
                print_json(json.dumps(response_data, indent=4))
                self.verification_token = response_data['verification_token']
                return response_data

    async def signup(self):
        url = 'https://user.mypikpak.com/v1/auth/signup'
        body = {
            "email": self.mail,
            "verification_code": self.code,
            "verification_token": self.verification_token,
            "password": "pw123456",
            "client_id": "YvtoWO6GNHiuCl7x"
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'accept-language': 'zh-CN',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('注册结果:')
                print_json(json.dumps(response_data, indent=4))
                self.sub = response_data['sub']
                self.access_token = response_data['access_token']
                return response_data

    async def login_init(self):
        url = 'https://user.mypikpak.com/v1/shield/captcha/init'
        body = {
            "client_id": "YvtoWO6GNHiuCl7x",
            "action": "POST:/v1/auth/signin",
            "device_id": self.xid,
            "captcha_token": '',
            "meta": {
                "email": self.mail
            },
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('登录安全验证:')
                print_json(json.dumps(response_data, indent=4))
                self.captcha_token = response_data['captcha_token']
                return response_data

    async def signin(self):
        url = 'https://user.mypikpak.com/v1/auth/signin'
        body = {"username": self.mail,
                "password": "pw123456",
                "client_id": "YvtoWO6GNHiuCl7x"}
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-captcha-token': self.captcha_token,
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('登录结果:')
                print_json(json.dumps(response_data, indent=4))
                self.access_token = response_data['access_token']
                return response_data

    async def init1(self):
        url = 'https://user.mypikpak.com/v1/shield/captcha/init'
        body = {
            "client_id": "YvtoWO6GNHiuCl7x",
            "action": "POST:/vip/v1/activity/invite",
            "device_id": self.xid,
            "captcha_token": self.access_token,
            "meta": {
                "captcha_sign": "1." + self.sign,
                "client_version": "undefined",
                "package_name": "mypikpak.com",
                "user_id": self.sub,
                "timestamp": self.t
            },
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('二次安全验证:')
                print_json(json.dumps(response_data, indent=4))
                self.captcha_token = response_data['captcha_token']
                return response_data

    async def invite(self):
        url = 'https://api-drive.mypikpak.com/vip/v1/activity/invite'
        body = {
            "apk_extra": {
                "invite_code": ""
            }
        }
        headers = {
            'host': 'api-drive.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'authorization': 'Bearer ' + self.access_token,
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250'
                          'Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-captcha-token': self.captcha_token,
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-system-language': 'zh-CN',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('确认邀请:')
                print_json(json.dumps(response_data, indent=4))
                return response_data

    async def init2(self):
        url = 'https://user.mypikpak.com/v1/shield/captcha/init'
        body = {
            "client_id": "YvtoWO6GNHiuCl7x",
            "action": "post:/vip/v1/order/activation-code",
            "device_id": self.xid,
            "captcha_token": self.access_token,
            "meta": {
                "captcha_sign": "1." + self.sign,
                "client_version": "undefined",
                "package_name": "mypikpak.com",
                "user_id": self.sub,
                "timestamp": self.t
            },
        }
        headers = {
            'host': 'user.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'MainWindow Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250 Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-device-model': 'electron%2F18.3.15',
            'x-device-name': 'PC-Electron',
            'x-device-sign': f'wdi10.{self.xid}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'x-net-work-type': 'NONE',
            'x-os-version': 'Win32',
            'x-platform-version': '1',
            'x-protocol-version': '301',
            'x-provider-name': 'NONE',
            'x-sdk-version': '7.0.7',
            'x-forwarded-for': self.IP
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('三次安全验证:')
                print_json(json.dumps(response_data, indent=4))
                return response_data

    async def activation_code(self):
        url = 'https://api-drive.mypikpak.com/vip/v1/order/activation-code'
        body = {
            "activation_code": self.incode,
            "page": "invite"
        }
        headers = {
            'host': 'api-drive.mypikpak.com',
            'content-length': str(len(json.dumps(body))),
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN',
            'authorization': 'Bearer ' + self.access_token,
            'referer': 'https://pc.mypikpak.com',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'PikPak/2.4.2.4250'
                          'Chrome/100.0.4896.160 Electron/18.3.15 Safari/537.36',
            'content-type': 'application/json',
            'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'x-captcha-token': self.captcha_token,
            'x-client-id': 'YvtoWO6GNHiuCl7x',
            'x-client-version': '2.4.2.4250',
            'x-device-id': self.xid,
            'x-system-language': 'zh-CN',
            'x-forwarded-for': self.IP
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, ssl=False) as response:
                response_data = await response.json()
                print('填写邀请:')
                print_json(json.dumps(response_data, indent=4))
                return response_data


# 主函数
async def main(in_code: str) -> AsyncGenerator[str, None]:
    try:
        pikpak = PIKPAK(incode=in_code)
        start_time = int(time.time())
        yield await pikpak.get_mail()
        yield await pikpak.init()
        print('注册滑块验证:...')
        while True:
            yield await pikpak.get_image()
            Verify = await pikpak.get_verify()
            if Verify['result'] == 'accept':
                print(f'验证通过!!!')
                break
            else:
                print('验证失败，重新验证...')
        yield await pikpak.get_new_token()
        yield await pikpak.verification()
        yield await pikpak.get_code()
        yield await pikpak.verify()
        yield await pikpak.signup()
        yield await pikpak.login_init()
        print('登录滑块验证:...')
        while True:
            yield await pikpak.get_image()
            Verify = await pikpak.get_verify()
            if Verify['result'] == 'accept':
                print(f'验证通过!!!')
                break
            else:
                print('验证失败，重新验证...')
        yield await pikpak.get_new_token()
        yield await pikpak.signin()
        yield await pikpak.init1()
        yield await pikpak.invite()
        yield await pikpak.init2()
        activation = await pikpak.activation_code()
        end_time = int(time.time())
        run_time = f'{(end_time - start_time):.2f}'
        if activation['add_days'] == 5:
            print(f'邀请码: {in_code} => 邀请成功, 运行时间: {run_time} 秒')
            yield f'邀请码: {in_code} => 邀请成功, \n运行时间: {run_time} 秒'
        elif activation['add_days'] == 0:
            print(f'邀请码: {in_code} => 邀请失败, 运行时间: {run_time} 秒')
            print(f'邀请码: {in_code} => 邀请失败, \n运行时间: {run_time} 秒')
        else:
            print(f'程序异常请重试!!!, 运行时间: {run_time} 秒')
    except Exception as e:
        print('异常捕获:', e)
        yield '结果:', e
        yield '用时:', run_time, '秒'


app = Quart(__name__)


@app.route('/')
async def index() -> Any:
    return await render_template('111.html')


@app.route('/api/run-script', methods=['POST'])
async def run_script() -> Any:
    try:
        data = await request.json
        in_code = data.get('invite_code')

        async def stream() -> AsyncGenerator[str, None]:
            async for item in main(in_code):
                yield item

        # Collect all items from the async generator into a list
        items = []
        async for item in stream():
            items.append(item)

        # Prepare response content
        content = '\n'.join(str(item) for item in items)

        return Response(content, content_type='text/plain')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10003)


