"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Flush the KV cache.

Usage:
python3 -m sglang.srt.mem_cache.flush_cache --url http://localhost:30000
"""

# 此文件提供一个命令行工具，用于远程刷新 SGLang 服务器的 KV 缓存。
# 通过向服务器的 /flush_cache 端点发送 HTTP POST 请求来触发缓存清除。
# 典型使用场景：调试时需要重置缓存状态，或在基准测试前清空前缀缓存。

import argparse

import requests

if __name__ == "__main__":
    # 解析命令行参数：--url 指定 SGLang 服务器地址，默认为本地 30000 端口
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:30000")
    args = parser.parse_args()

    # 向服务器发送缓存刷新请求，断言返回状态码为 200 表示成功
    response = requests.post(args.url + "/flush_cache")
    assert response.status_code == 200
