import ast
import json
import random
import re
import time

#from openai import OpenAI
#from tqdm import tqdm

# --- 配置 ---
API_KEY = "sk-ce04efb46aac4b12bbc90208c7ae27a3"  
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
MODEL_NAME = "deepseek-v3.2-exp"

PROBLEM_FILE = 'problem.json' 
STUDENT_FILE = 'student-problem-fine.json' 
TEST_SAMPLE_SIZE = 200
SHOW_PROMPT_SAMPLES = True  # 是否生成并显示prompt示例
NUM_PROMPT_SAMPLES = 3  # 生成多少个prompt示例 

#client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 检索器 ---
class RandomRetriever:
    def retrieve(self, history, target_prob, k=4):
        candidates = [h for h in history if h['problem_id'] != target_prob['problem_id']]
        if len(candidates) < k: return candidates
        return random.sample(candidates, k)

class CognitiveRetriever:
    def __init__(self, collaborative_file='item_collaborative.json', collaborative_data=None):
        """
        初始化检索器，加载协同相似项数据
        
        Args:
            collaborative_file: 协同数据文件路径（如果未提供 collaborative_data）
            collaborative_data: 已加载的协同数据字典（如果提供，优先使用）
        """
        if collaborative_data is not None:
            self.collaborative_similar = collaborative_data
        else:
            self.collaborative_similar = {}
            try:
                with open(collaborative_file, 'r', encoding='utf-8') as f:
                    self.collaborative_similar = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {collaborative_file} not found, collaborative signals will be disabled.")
            except Exception as e:
                print(f"Warning: Failed to load {collaborative_file}: {e}")
    
   def retrieve(self, history, target_prob, k=4):
        candidates = []
        # ... (获取 target_id, concepts, level 等代码不变)
        
        # 定义动态权重的基准参数
        W_CF_HIGH = 8  # 认知断层时，协同权重提升（甚至可以高于 wp）
        W_CF_LOW = 2   # 认知充足时，协同权重降低（作为辅助）
        
        for h in history:
            h_id = h['problem_id']
            if h_id not in problems: continue
            h_meta = problems[h_id]
            
            # 分离计算认知得分 (Cognitive Score)
            cog_score = 0
            # 1. 知识一致性 (wc=10)
            h_concepts = set(h_meta.get('concepts', []))
            if not target_concepts.isdisjoint(h_concepts): cog_score += 10
            
            # 2. 认知层级 (wp=5, wm=2)
            h_level = h_meta.get('cognitive_dimension', 0)
            if h_level > 0 and target_level > 0:
                if h_level < target_level: cog_score += 5
                if h_level == target_level: cog_score += 2
            
            # 3. 动态协同关联 (Dynamic wcf)
            final_score = cog_score
            if h_id in target_similar_items:
                # 动态逻辑：如果认知得分低（例如为0，说明完全没有知识点或层级关联），则提升协同权重
                # 这符合文档中 "explicit cognitive retrieval... below threshold... automatic increase wcf"
                if cog_score < 5: 
                    final_score += W_CF_HIGH 
                else:
                    final_score += W_CF_LOW
            
            if final_score > 0: candidates.append((final_score, h))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]

# --- 辅助函数 ---
def extract_content(problem_record):
    """从问题记录中提取 content 文本"""
    detail = problem_record.get('detail', '')
    if not detail:
        return ''
    try:
        if isinstance(detail, str):
            d = ast.literal_eval(detail)
            if isinstance(d, dict):
                return d.get('content', '') or ''
        elif isinstance(detail, dict):
            return detail.get('content', '') or ''
    except Exception:
        pass
    return ''

# --- Prompt 生成器 ---
def generate_prompt(retrieved, target, history_full, semantic_map, collab_map):
    """
    生成符合 Cognitive-RAG 论文格式的 Prompt
    
    Args:
        retrieved: 检索到的 Top-K 题目列表（来自 retrieve 方法）
        target: 目标题目的元数据（problems[target_id]）
        history_full: 完整的历史序列（用于生成 USER TRAJECTORY）
        semantic_map: {problem_id: semantic_id} 映射
        collab_map: {problem_id: [similar_problem_ids]} 映射
    """
    
    # [USER TRAJECTORY (Coarse-grained)]: 最近10条交互的语义ID和Result
    trajectory_lines = []
    recent_history = history_full[-10:] if len(history_full) >= 10 else history_full
    for h in recent_history:
        pid = h.get('problem_id', '')
        if not pid:
            continue
        semantic_id = semantic_map.get(pid, 'Unknown')
        result_str = "Correct" if h.get('is_correct', 0) > 0 else "Incorrect"
        trajectory_lines.append(f"  {semantic_id}: {result_str}")
    
    trajectory_str = "\n".join(trajectory_lines) if trajectory_lines else "  (No history)"
    
    # [RETRIEVED CONTEXT (Fine-grained)]: Top-K 题目的详细信息
    retrieved_lines = []
    for r in retrieved:
        pid = r.get('problem_id', '')
        if pid not in problems:
            continue
        meta = problems[pid]
        
        # Question ID: 分层语义ID
        semantic_id = semantic_map.get(pid, 'Unknown')
        
        # Collaborative: 展示该题目的 3 个相似题的语义ID字符串
        similar_pids = collab_map.get(pid, [])[:3]
        similar_semantic_ids = [semantic_map.get(sid, 'Unknown') for sid in similar_pids]
        collaborative_str = ", ".join(similar_semantic_ids) if similar_semantic_ids else "None"
        
        # Cognitive: 知识点和认知层级
        concepts_list = meta.get('concepts', [])
        concepts_str = ", ".join(concepts_list) if concepts_list else "None"
        cognitive_level = meta.get('cognitive_dimension', 'Unknown')
        cognitive_str = f"Concepts: {concepts_str}, Level: {cognitive_level}"
        
        # Content: 题目文本摘要
        content = extract_content(meta)
        content_str = content[:200] if content else "N/A"  # 限制长度
        
        # Result: 学生在该题的作答结果
        result_str = "Correct" if r.get('is_correct', 0) > 0 else "Incorrect"
        
        retrieved_lines.append(f"  Question ID: {semantic_id}")
        retrieved_lines.append(f"  Collaborative: {collaborative_str}")
        retrieved_lines.append(f"  Cognitive: {cognitive_str}")
        retrieved_lines.append(f"  Content: {content_str}")
        retrieved_lines.append(f"  Result: {result_str}")
        retrieved_lines.append("")  # 空行分隔
    
    retrieved_str = "\n".join(retrieved_lines).strip()
    
    # [TARGET PROBLEM]: 目标题目信息（不含 Result）
    target_id = target.get('problem_id', '')
    target_semantic_id = semantic_map.get(target_id, 'Unknown')
    target_similar_pids = collab_map.get(target_id, [])[:3]
    target_similar_semantic_ids = [semantic_map.get(sid, 'Unknown') for sid in target_similar_pids]
    target_collaborative_str = ", ".join(target_similar_semantic_ids) if target_similar_semantic_ids else "None"
    
    target_concepts_list = target.get('concepts', [])
    target_concepts_str = ", ".join(target_concepts_list) if target_concepts_list else "None"
    target_cognitive_level = target.get('cognitive_dimension', 'Unknown')
    target_cognitive_str = f"Concepts: {target_concepts_str}, Level: {target_cognitive_level}"
    
    target_content = extract_content(target)
    target_content_str = target_content[:200] if target_content else "N/A"
    
    target_str = f"""  Question ID: {target_semantic_id}
  Collaborative: {target_collaborative_str}
  Cognitive: {target_cognitive_str}
  Content: {target_content_str}"""
    
    # 组装完整 Prompt
    prompt = f"""{system_role}

[USER TRAJECTORY (Coarse-grained)]:
{trajectory_str}

[RETRIEVED CONTEXT (Fine-grained)]:
{retrieved_str}

[TARGET PROBLEM]:
{target_str}

{instruction}"""
    
    return prompt

def get_llm_prediction(prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # time.sleep(2) 
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5 # 强制只吐数字
            )
            content = response.choices[0].message.content.strip()
            # 暴力解析
            if "1" in content: return 1
            if "0" in content: return 0
            return 0
        except Exception as e:
            if "429" in str(e): time.sleep(10)
    return 0

# --- 主程序 ---
print("Loading data...")
problems = {}
with open(PROBLEM_FILE, 'r', encoding='utf-8') as f:
    try:
        for line in f:
            if line.strip():
                p = json.loads(line)
                problems[p['problem_id']] = p
    except:
        f.seek(0); data = json.load(f)
        for p in data: problems[p['problem_id']] = p

raw_data = []
with open(STUDENT_FILE, 'r', encoding='utf-8') as f:
    try: raw_data = json.load(f)
    except: f.seek(0); raw_data = [json.loads(line) for line in f if line.strip()]

# 加载语义ID和协同相似项映射
print("Loading semantic and collaborative maps...")
semantic_map = {}
with open('item_semantic_ids.json', 'r', encoding='utf-8') as f:
    semantic_map = json.load(f)

collab_map = {}
with open('item_collaborative.json', 'r', encoding='utf-8') as f:
    collab_map = json.load(f)

print(f"准备测试样本 (N={TEST_SAMPLE_SIZE})...")
random.seed(42)
valid_indices = [i for i, s in enumerate(raw_data) if len(s['seq']) >= 5]
test_indices = random.sample(valid_indices, min(TEST_SAMPLE_SIZE, len(valid_indices)))

# 注释：暂时不进行推理，只查看prompt构建效果
# results = {'random': [], 'ours_old': [], 'ours_new': [], 'ground_truth': []}
random_retriever = RandomRetriever()
# 实例化 CognitiveRetriever 时传入已加载的协同数据（避免重复加载）
cognitive_retriever = CognitiveRetriever(collaborative_data=collab_map)

# --- 生成并查看Prompt示例 ---
if SHOW_PROMPT_SAMPLES:
    print("\n" + "="*60)
    print("生成Prompt构建示例...")
    print("="*60)
    
    sample_indices = test_indices[:NUM_PROMPT_SAMPLES]
    prompt_samples = []
    
    for sample_idx, idx in enumerate(sample_indices, 1):
        student = raw_data[idx]
        seq = student['seq']
        target = seq[-1]
        history = seq[:-1]
        if target['problem_id'] not in problems: continue
        target_meta = problems[target['problem_id']]
        
        # 生成两种检索方式的prompt
        r_ctx = random_retriever.retrieve(history, target_meta)
        c_ctx = cognitive_retriever.retrieve(history, target_meta)
        
        prompt_random = generate_prompt(r_ctx, target_meta, history, semantic_map, collab_map)
        prompt_cognitive = generate_prompt(c_ctx, target_meta, history, semantic_map, collab_map)
        
        sample_info = {
            'sample_index': sample_idx,
            'student_index': idx,
            'target_problem_id': target['problem_id'],
            'target_semantic_id': semantic_map.get(target['problem_id'], 'Unknown'),
            'history_length': len(history),
            'random_retrieved_count': len(r_ctx),
            'cognitive_retrieved_count': len(c_ctx),
            'ground_truth': 1 if target.get('is_correct', 0) > 0 else 0,
            'prompt_random': prompt_random,
            'prompt_cognitive': prompt_cognitive
        }
        prompt_samples.append(sample_info)
        
        # 打印到控制台
        print(f"\n【示例 {sample_idx}】")
        print(f"目标题目ID: {target['problem_id']}")
        print(f"目标题目语义ID: {semantic_map.get(target['problem_id'], 'Unknown')}")
        print(f"历史题目数量: {len(history)}")
        print(f"随机检索数量: {len(r_ctx)}, 认知检索数量: {len(c_ctx)}")
        print(f"真实答案: {'正确' if sample_info['ground_truth'] == 1 else '错误'}")
        print("\n--- Random Baseline Prompt (前500字符) ---")
        print(prompt_random[:500] + "..." if len(prompt_random) > 500 else prompt_random)
        print("\n--- Cognitive-RAG Prompt (前500字符) ---")
        print(prompt_cognitive[:500] + "..." if len(prompt_cognitive) > 500 else prompt_cognitive)
        print("-" * 60)
    
    # 保存完整prompt到文件
    output_file = 'prompt_samples.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompt_samples, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 完整Prompt示例已保存到: {output_file}")
    
    # 同时保存可读的文本版本
    output_txt_file = 'prompt_samples.txt'
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for sample in prompt_samples:
            f.write("="*80 + "\n")
            f.write(f"示例 {sample['sample_index']}\n")
            f.write("="*80 + "\n")
            f.write(f"目标题目ID: {sample['target_problem_id']}\n")
            f.write(f"目标题目语义ID: {sample['target_semantic_id']}\n")
            f.write(f"历史题目数量: {sample['history_length']}\n")
            f.write(f"随机检索数量: {sample['random_retrieved_count']}\n")
            f.write(f"认知检索数量: {sample['cognitive_retrieved_count']}\n")
            f.write(f"真实答案: {'正确' if sample['ground_truth'] == 1 else '错误'}\n\n")
            
            f.write("\n【Random Baseline Prompt】\n")
            f.write("-"*80 + "\n")
            f.write(sample['prompt_random'])
            f.write("\n\n")
            
            f.write("\n【Cognitive-RAG Prompt】\n")
            f.write("-"*80 + "\n")
            f.write(sample['prompt_cognitive'])
            f.write("\n\n\n")
    print(f"✓ 可读文本版本已保存到: {output_txt_file}")
    print("="*60 + "\n")

# --- 暂时注释掉大模型推理部分，只查看prompt构建效果 ---
# for idx in tqdm(test_indices):
#     student = raw_data[idx]
#     seq = student['seq']
#     target = seq[-1]
#     history = seq[:-1]
#     if target['problem_id'] not in problems: continue
#     target_meta = problems[target['problem_id']]
#     gt = 1 if target.get('is_correct', 0) > 0 else 0
#     
#     # 1. Random Baseline
#     r_ctx = random_retriever.retrieve(history, target_meta)
#     results['random'].append(get_llm_prediction(generate_prompt(r_ctx, target_meta, history, semantic_map, collab_map)))
#     
#     # 2. Ours Old (Complex Prompt) - 复现之前的 60%
#     # c_ctx = cognitive_retriever.retrieve(history, target_meta)
#     # results['ours_old'].append(get_llm_prediction(generate_prompt(c_ctx, target_meta, history, semantic_map, collab_map)))
#     
#     # 3. Ours New (Cognitive-RAG Prompt) - 核心实验组
#     c_ctx = cognitive_retriever.retrieve(history, target_meta)
#     results['ours_new'].append(get_llm_prediction(generate_prompt(c_ctx, target_meta, history, semantic_map, collab_map)))
#     
#     results['ground_truth'].append(gt)
# 
# # --- 统计 ---
# from sklearn.metrics import accuracy_score
# 
# y_true = results['ground_truth']
# acc_rand = accuracy_score(y_true, results['random'])
# acc_new = accuracy_score(y_true, results['ours_new'])
# 
# print("\n" + "="*40)
# print(f"ABLATION RESULTS (N={len(y_true)})")
# print(f"Random Baseline (Simple): {acc_rand:.4f}")
# print(f"Ours New (Simple Prompt): {acc_new:.4f} ")
# print(f"Improvement: +{(acc_new - acc_rand)*100:.2f}%")
# print("="*40)

print("\n" + "="*60)
print("提示：已跳过大模型推理步骤，只查看prompt构建效果")
print("如需进行完整实验，请取消注释主循环和统计部分的代码")
print("="*60)