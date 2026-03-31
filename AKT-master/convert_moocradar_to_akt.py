# convert_moocradar_to_akt.py
import json
import numpy as np
from collections import defaultdict
import os

def load_moocradar_data(json_path):
    """加载MOOCRadar JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_moocradar_data(data, problem_detail_path=None):
    """
    处理MOOCRadar数据，转换为AKT格式
    """
    # 1. 收集所有问题和答案
    all_problems = set()
    all_students = []
    
    for student_data in data:
        seq = student_data.get('seq', [])
        if not seq:
            continue
            
        student_seq = []
        for item in seq:
            problem_id = item.get('problem_id')
            is_correct = item.get('is_correct', 0)
            
            if problem_id:
                all_problems.add(problem_id)
                student_seq.append({
                    'problem_id': problem_id,
                    'is_correct': int(is_correct)
                })
        
        if student_seq:
            all_students.append(student_seq)
    
    # 2. 创建问题ID到编号的映射
    # problem_id -> problem_index (从1开始，因为0通常用于padding)
    problem_list = sorted(list(all_problems))
    problem_to_index = {pid: idx + 1 for idx, pid in enumerate(problem_list)}
    num_questions = len(problem_list)
    
    print(f"总问题数: {num_questions}")
    print(f"总学生数: {len(all_students)}")
    
    # 3. 为PID模式创建映射（如果problem_id就是PID，那么两者相同）
    # 但为了区分概念和具体问题，我们使用不同的映射
    pid_list = sorted(list(all_problems))  # 这里可以用exercise_id或其他作为PID
    pid_to_index = {pid: idx + 1 for idx, pid in enumerate(pid_list)}
    num_pids = len(pid_list)
    
    return all_students, problem_to_index, pid_to_index, num_questions, num_pids

def convert_to_akt_format(all_students, problem_to_index, pid_to_index, output_path):
    """
    将数据转换为AKT需要的CSV格式（PID模式，4行一个学生）
    确保P、Q、A序列长度完全一致
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for student_idx, student_seq in enumerate(all_students):
            if not student_seq:
                continue
            
            # 确保所有序列长度一致
            pid_seq = [str(pid_to_index[item['problem_id']]) for item in student_seq]
            q_seq = [str(problem_to_index[item['problem_id']]) for item in student_seq]
            a_seq = [str(item['is_correct']) for item in student_seq]
            
            # 验证长度一致性
            assert len(pid_seq) == len(q_seq) == len(a_seq), \
                f"学生 {student_idx} 的序列长度不一致: PID={len(pid_seq)}, Q={len(q_seq)}, A={len(a_seq)}"
            
            # 第一行：学生ID
            f.write(f"{student_idx}\n")
            # 第二行：PID序列
            f.write(','.join(pid_seq) + '\n')
            # 第三行：Q序列
            f.write(','.join(q_seq) + '\n')
            # 第四行：答案序列
            f.write(','.join(a_seq) + '\n')

def split_train_valid_test(all_students, train_ratio=0.8, valid_ratio=0.1):
    """
    划分训练集、验证集和测试集
    """
    np.random.seed(224)  # 与AKT代码保持一致
    np.random.shuffle(all_students)
    
    total = len(all_students)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))
    
    train_data = all_students[:train_end]
    valid_data = all_students[train_end:valid_end]
    test_data = all_students[valid_end:]
    
    print(f"训练集: {len(train_data)} 学生")
    print(f"验证集: {len(valid_data)} 学生")
    print(f"测试集: {len(test_data)} 学生")
    
    return train_data, valid_data, test_data

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据路径（基于脚本目录）
    data_dir = os.path.join(script_dir, 'data', 'MOOCRadar')
    json_path = os.path.join(data_dir, 'student-problem-fine.json')
    problem_detail_path = os.path.join(data_dir, 'problem.json')
    output_dir = os.path.join(data_dir, 'akt_format')
    
    # 检查输入文件是否存在
    if not os.path.exists(json_path):
        print(f"错误：找不到文件 {json_path}")
        print(f"请确保文件存在")
        exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载MOOCRadar数据...")
    print(f"数据文件: {json_path}")
    data = load_moocradar_data(json_path)
    
    # 处理数据
    print("处理数据...")
    all_students, problem_to_index, pid_to_index, num_questions, num_pids = process_moocradar_data(
        data, problem_detail_path
    )
    
    # 划分数据集
    print("划分数据集...")
    train_data, valid_data, test_data = split_train_valid_test(all_students)
    
    # 转换为AKT格式
    print("转换为AKT格式...")
    convert_to_akt_format(train_data, problem_to_index, pid_to_index, 
                         os.path.join(output_dir, 'moocradar_pid_train1.csv'))  # 添加_pid
    convert_to_akt_format(valid_data, problem_to_index, pid_to_index, 
                         os.path.join(output_dir, 'moocradar_pid_valid1.csv'))  # 添加_pid
    convert_to_akt_format(test_data, problem_to_index, pid_to_index, 
                         os.path.join(output_dir, 'moocradar_pid_test1.csv'))  # 添加_pid
    
    # 保存统计信息
    stats = {
        'num_questions': num_questions,
        'num_pids': num_pids,
        'num_train_students': len(train_data),
        'num_valid_students': len(valid_data),
        'num_test_students': len(test_data)
    }
    
    with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据转换完成！")
    print(f"统计信息: {stats}")
    print(f"\n输出目录: {output_dir}")
    print(f"\n下一步：在main.py中使用统计信息更新参数")

if __name__ == '__main__':
    main()