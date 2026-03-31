import json
import random

# 1. 读取生成的结果
print("Loading semantic IDs...")
with open('item_semantic_ids.json', 'r', encoding='utf-8') as f:
    semantic_data = json.load(f)

# 2. 读取题目文本 (注意这里针对你的数据结构做了修改)
print("Loading problem content...")
pid_to_content = {}
with open('problem.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            p = json.loads(line)
            pid = str(p.get('problem_id', p.get('id', 'unknown')))
            
            # --- 核心修改：从 detail 中提取 content ---
            detail = p.get('detail', {})
            # detail 可能是字典，也可能是字符串，这里做个兼容处理
            if isinstance(detail, str):
                try:
                    # 尝试把字符串解析为 JSON，如果失败则保留原样
                    # 注意：如果字符串里是单引号（Python dict格式），json.loads会失败
                    # 这里假设是标准JSON或者直接跳过复杂解析，先试简单的
                    detail = json.loads(detail) 
                except:
                    pass 
            
            if isinstance(detail, dict):
                content = detail.get('content', 'No Content Key in Detail')
            else:
                content = str(detail) # 如果实在解析不了，就看 detail 本身
            # ----------------------------------------

            pid_to_content[pid] = content
        except json.JSONDecodeError:
            continue

# 3. 统计每个 cluster 的题目
clusters = {}
for pid, sem_id in semantic_data.items():
    if sem_id not in clusters:
        clusters[sem_id] = []
    clusters[sem_id].append(pid)

# 4. 打印结果
sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
print(f"Total Clusters: {len(sorted_clusters)}")
print("-" * 30)

print("【最大的 3 个簇】")
for sem_id, pids in sorted_clusters[:3]:
    percentage = (len(pids) / len(semantic_data)) * 100
    print(f"\nCluster ID: {sem_id} (Count: {len(pids)} | {percentage:.2f}%)")
    
    sample_pids = random.sample(pids, min(3, len(pids)))
    for pid in sample_pids:
        text = str(pid_to_content.get(pid, 'No Content Found'))
        # 清理 HTML 标签（简单粗暴版），只看前 100 字
        clean_text = text.replace('\n', ' ').replace('<p>', '').replace('</p>', '')[:100]
        print(f"  - [PID {pid}]: {clean_text}...")

print("-" * 30)