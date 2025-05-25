import os
import shutil
import json

# 输入源目录与目标子目录的映射关系
source_dirs = {
    'us_train2': {'low_quality': 'trainA', 'high_quality': 'trainB'},
    'us_test2': {'low_quality': 'testA',  'high_quality': 'testB'}
}

# 所有类别名和对应数字的映射
class_names = set()

# 先收集所有类别名
for base_dir in source_dirs:
    if not os.path.exists(base_dir):
        continue
    for class_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, class_name)):
            class_names.add(class_name)

# 排序并从 1 开始编号
class_to_index = {name: idx + 1 for idx, name in enumerate(sorted(class_names))}

# 显示映射（可选）
print("类别映射：", class_to_index)

# 用于存储标签信息
labels = {}

# 遍历两个源目录（train 和 test）
for base_dir, quality_map in source_dirs.items():
    if not os.path.exists(base_dir):
        continue

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for quality_type, target_dir in quality_map.items():
            src_quality_dir = os.path.join(class_path, quality_type)
            if not os.path.isdir(src_quality_dir):
                continue

            print(target_dir)
            os.makedirs(target_dir, exist_ok=True)

            for fname in os.listdir(src_quality_dir):
                src_file = os.path.join(src_quality_dir, fname)
                if os.path.isfile(src_file):
                    dst_file = os.path.join(target_dir, fname)
                    shutil.copy2(src_file, dst_file)

                    # 从文件名中提取编号（去除扩展名）
                    image_id = os.path.splitext(fname)[0]
                    labels[image_id] = class_to_index[class_name]

# 保存为 label.json
with open('label.json', 'w') as f:
    json.dump(labels, f, indent=2)

print("图像整理完毕，标签已保存到 label.json")
