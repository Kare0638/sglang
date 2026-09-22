"""审阅修改后的真机验证：每条对应一个具体意见或一个必须保持的行为。"""
import json, urllib.request, urllib.error

URL = "http://127.0.0.1:30000/generate"
RESULTS = []


def post(body, stream=False):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = r.read().decode()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'✅' if ok else '❌'} {name}" + (f"   ← {detail}" if detail else ""))


def sse_indexes(raw):
    idx, frames = set(), 0
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            frames += 1
            d = json.loads(line[6:])
            if "index" in d:
                idx.add(d["index"])
    return frames, idx


SP = {"temperature": 0.8, "max_new_tokens": 12}

print("\n[必须保持的行为]")
st, raw = post({"text": "The capital of France is", "sampling_params": {**SP, "n": 3}})
d = json.loads(raw)
ok = st == 200 and isinstance(d, list) and len(d) == 3
check("n=3 → 长度 3 的数组", ok, f"HTTP {st}, {type(d).__name__}")
if ok:
    texts = [x["text"] for x in d]
    check("三段输出互不相同（真采样，不是复制）", len(set(texts)) == 3)
    ids = [x["meta_info"]["id"] for x in d]
    check("三个 meta_info.id 互不相同", len(set(ids)) == 3, ", ".join(ids))

st, raw = post({"text": "The capital of France is", "sampling_params": {**SP, "n": 1}})
d = json.loads(raw)
check("n=1 → 单个对象（不是 [{...}]）", st == 200 and isinstance(d, dict), f"{type(d).__name__}")

st, raw = post({"text": ["Red is", "Water is"], "sampling_params": {**SP, "n": 2}})
d = json.loads(raw)
check("2 prompts × n=2 → 4 项", st == 200 and isinstance(d, list) and len(d) == 4)

st, raw = post({"text": "One two", "stream": True, "sampling_params": {**SP, "n": 3}})
frames, idx = sse_indexes(raw)
check("流式 n=3 → index 集合 {0,1,2}", idx == {0, 1, 2}, f"{frames} 帧, index={sorted(idx)}")

st, raw = post({"text": "One two", "stream": True, "sampling_params": {**SP, "n": 1}})
frames, idx = sse_indexes(raw)
check("流式 n=1 → 不带 index", not idx, f"{frames} 帧")

print("\n[意见⑥ 多模态：现在任何模型都拒]")
st, raw = post({"text": "a", "image_data": "https://example.com/x.png", "sampling_params": {**SP, "n": 3}})
# 必须断言消息：守卫若失效，请求会走到下载图片那一步，下载失败同样是 400——那就是假通过。
check("纯文本模型 + image_data + n=3 → 400，且是守卫拒的（旧行为是放行）",
      st == 400 and "multimodal fields" in raw, raw[:110])

print("\n[意见⑦ rid 加尾巴后的长度上限（128）]")
# n=11 最多追加 "_10" 共 3 字节
st, raw = post({"text": "a", "rid": "x" * 125, "sampling_params": {**SP, "max_new_tokens": 2, "n": 11}})
check("rid 125 字节 + n=11（加尾巴后正好 128）→ 200", st == 200, f"HTTP {st}")
st, raw = post({"text": "a", "rid": "x" * 126, "sampling_params": {**SP, "max_new_tokens": 2, "n": 11}})
check("rid 126 字节 + n=11（加尾巴后 129）→ 400，且说明原因", st == 400 and "each sample's rid gains" in raw, raw[:120])

print("\n[提前校验：一个坏参数 = 一个 400]")
st, raw = post({"text": "a", "sampling_params": {"n": 3, "top_p": -1}})
check("n=3 + top_p=-1 → 单个 400（不是 200+数组）", st == 400 and not raw.lstrip().startswith("["), raw[:100])

print("\n[意见① 数量上限]")
st, raw = post({"text": "a", "sampling_params": {"n": 9223372036854775807}})
check("n=i64::MAX → 400，不 panic", st == 400, raw[:90])

print(f"\n合计: {sum(RESULTS)}/{len(RESULTS)} 通过")
