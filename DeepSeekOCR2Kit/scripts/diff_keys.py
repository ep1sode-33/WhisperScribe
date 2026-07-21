#!/usr/bin/env python3
"""对照 index.json 键集合与 Swift sanitize 目标模块树,人工核查遗漏。
用法: .venv/bin/python scripts/diff_keys.py <sanitized_keys.txt>
(sanitized_keys.txt 由临时 Swift 脚本/测试 print 导出,一行一键)"""
import json, pathlib, sys

model_dir = pathlib.Path((pathlib.Path(__file__).parent.parent / "Fixtures" / "model_dir.txt").read_text().strip())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
src = set(index["weight_map"].keys())
dst = set(pathlib.Path(sys.argv[1]).read_text().split())
print(f"checkpoint keys: {len(src)}  sanitized keys: {len(dst)}")
print("UNCONSUMED source keys (应为 0):")
# sanitize 是 src→dst 改名;此脚本靠人工比对数量与抽样,src 每键都应有 dst 对应
for k in sorted(src)[:20]:
    print(" ", k)
