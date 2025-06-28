import os
import shutil
from pathlib import Path

def copy_files_by_type(txt_file, file_mappings, target_base_dir):
    """
    根据txt文件中的文件名列表，复制不同类型的文件到统一目录
    
    Args:
        txt_file: 包含文件名的txt文件路径
        file_mappings: 文件类型映射字典，格式：{文件类型: (源目录, 文件扩展名)}
        target_base_dir: 目标基础目录
    """
    print(f"\n处理文件列表: {txt_file}")
    print("="*60)
    
    # 读取文件名列表
    try:
        with open(txt_file, 'r') as f:
            base_names = [os.path.splitext(line.strip())[0] for line in f if line.strip()]
        print(f"读取到 {len(base_names)} 个文件名")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_file}")
        return {}
    
    # 获取txt文件的基础名称（用于统计）
    txt_basename = Path(txt_file).stem
    
    # 统计信息
    stats = {file_type: {'success': 0, 'failed': 0, 'skipped': 0} for file_type in file_mappings.keys()}
    
    # 处理每种文件类型
    for file_type, (source_dir, extension) in file_mappings.items():
        print(f"\n处理 {file_type} 文件 (扩展名: {extension})")
        print("-" * 40)
        
        # 创建统一的目标目录（不按分割文件分组）
        target_dir = os.path.join(target_base_dir, file_type)
        os.makedirs(target_dir, exist_ok=True)
        print(f"目标目录: {target_dir}")
        
        # 检查源目录是否存在
        if not os.path.exists(source_dir):
            print(f"警告: 源目录不存在 {source_dir}")
            continue
        
        # 复制文件
        for filename in base_names:
            src_path = os.path.join(source_dir, filename + extension)
            
            # 对于NPY文件，目标文件名需要去掉_log_hdr
            if file_type == 'ldr_to_hdr_full_range_batch_full':
                dst_filename = filename + '.npy'
            else:
                dst_filename = filename + extension
            
            dst_path = os.path.join(target_dir, dst_filename)
            
            if os.path.isfile(src_path):
                # 检查目标文件是否已存在
                if os.path.exists(dst_path):
                    stats[file_type]['skipped'] += 1
                    if stats[file_type]['skipped'] <= 3:
                        print(f"⚠ 文件已存在，跳过: {dst_filename}")
                    elif stats[file_type]['skipped'] == 4:
                        print("  ... (更多文件已存在，跳过)")
                else:
                    try:
                        shutil.copy(src_path, dst_path)
                        stats[file_type]['success'] += 1
                        if stats[file_type]['success'] <= 5:  # 只显示前5个成功的文件
                            print(f"✓ 复制成功: {filename}{extension} -> {dst_filename}")
                        elif stats[file_type]['success'] == 6:
                            print("  ... (更多文件复制成功)")
                    except Exception as e:
                        print(f"✗ 复制失败: {filename}{extension} - {e}")
                        stats[file_type]['failed'] += 1
            else:
                stats[file_type]['failed'] += 1
                if stats[file_type]['failed'] <= 3:  # 只显示前3个失败的文件
                    print(f"✗ 文件不存在: {filename}{extension}")
                elif stats[file_type]['failed'] == 4:
                    print("  ... (更多文件不存在)")
    
    # 打印统计信息
    print(f"\n{txt_basename} 处理完成统计:")
    print("-" * 40)
    for file_type, stat in stats.items():
        total = stat['success'] + stat['failed'] + stat['skipped']
        if total > 0:
            success_rate = (stat['success'] / total * 100)
            print(f"{file_type:20}: 成功 {stat['success']:4d} | 失败 {stat['failed']:4d} | 跳过 {stat['skipped']:4d} | 成功率 {success_rate:5.1f}%")
    
    return stats

def main():
    """主函数"""
    print("批量文件复制工具 - 统一目录版本")
    print("="*60)
    
    # 基础路径配置
    base_path = '/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN'
    
    # 分割文件列表
    split_files = [
        'dgain_0_50.txt',
        'dgain_50_100.txt', 
        'dgain_100_150.txt',
        'dgain_150_200.txt'
    ]
    
    # 文件类型映射：{文件类型: (源目录, 文件扩展名)}
    file_mappings = {
        'dgain_info': (
            os.path.join(base_path, 'FULL_DATA/dgain_info'),
            '.txt'
        ),
        'hdr': (
            os.path.join(base_path, 'FULL_DATA/hdr'),
            '.hdr'
        ),
        'jpg': (
            os.path.join(base_path, 'FULL_DATA/jpg'),
            '.jpg'
        ),
        'ldr_to_hdr_full_range_batch_full': (
            os.path.join(base_path, 'FULL_DATA/ldr_to_hdr_full_range_batch_full'),
            '_log_hdr.npy'
        )
    }
    
    # 目标基础目录
    target_base_dir = os.path.join(base_path, 'merged_data')
    
    # 显示配置信息
    print("配置信息:")
    print(f"基础路径: {base_path}")
    print(f"目标基础目录: {target_base_dir}")
    print(f"分割文件: {split_files}")
    print("\n文件类型映射:")
    for file_type, (source_dir, ext) in file_mappings.items():
        print(f"  {file_type:30} -> {source_dir} (*{ext})")
    
    print(f"\n最终目录结构将是:")
    print(f"{target_base_dir}/")
    for file_type in file_mappings.keys():
        print(f"  ├── {file_type}/")
    
    # 确认是否继续
    user_input = input(f"\n是否继续处理? (y/n): ")
    if user_input.lower() != 'y':
        print("操作已取消")
        return
    
    # 总体统计
    total_stats = {file_type: {'success': 0, 'failed': 0, 'skipped': 0} for file_type in file_mappings.keys()}
    
    # 处理每个分割文件
    total_start_time = __import__('time').time()
    
    for split_file in split_files:
        txt_file_path = os.path.join(base_path, 'dgain_split_file', split_file)
        
        if not os.path.exists(txt_file_path):
            print(f"\n警告: 分割文件不存在 {txt_file_path}")
            continue
        
        start_time = __import__('time').time()
        file_stats = copy_files_by_type(txt_file_path, file_mappings, target_base_dir)
        end_time = __import__('time').time()
        
        # 累加统计
        for file_type in file_mappings.keys():
            if file_type in file_stats:
                total_stats[file_type]['success'] += file_stats[file_type]['success']
                total_stats[file_type]['failed'] += file_stats[file_type]['failed']
                total_stats[file_type]['skipped'] += file_stats[file_type]['skipped']
        
        print(f"处理时间: {end_time - start_time:.2f} 秒")
    
    total_end_time = __import__('time').time()
    
    # 显示总体统计
    print(f"\n" + "="*60)
    print("总体处理统计:")
    print("="*60)
    for file_type, stat in total_stats.items():
        total = stat['success'] + stat['failed'] + stat['skipped']
        if total > 0:
            success_rate = (stat['success'] / total * 100)
            print(f"{file_type:30}: 成功 {stat['success']:4d} | 失败 {stat['failed']:4d} | 跳过 {stat['skipped']:4d} | 成功率 {success_rate:5.1f}%")
    
    print(f"\n总处理时间: {total_end_time - total_start_time:.2f} 秒")
    print("所有文件处理完成!")
    
    # 显示最终目录结构和文件统计
    print(f"\n最终目录结构:")
    print("-" * 40)
    for file_type in file_mappings.keys():
        type_dir = os.path.join(target_base_dir, file_type)
        if os.path.exists(type_dir):
            file_count = len([f for f in os.listdir(type_dir) if os.path.isfile(os.path.join(type_dir, f))])
            print(f"{file_type}/ ({file_count} 文件)")

def create_summary_report(target_base_dir):
    """创建处理摘要报告"""
    report_file = os.path.join(target_base_dir, 'processing_summary.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("文件处理摘要报告 - 统一目录版本\n")
        f.write("="*50 + "\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("最终数据统计:\n")
        f.write("-" * 30 + "\n")
        
        total_files = 0
        for file_type in ['dgain_info', 'hdr', 'jpg', 'ldr_to_hdr_full_range_batch_full']:
            type_path = os.path.join(target_base_dir, file_type)
            if os.path.exists(type_path):
                file_count = len([f for f in os.listdir(type_path) if os.path.isfile(os.path.join(type_path, f))])
                f.write(f"  {file_type}: {file_count} 文件\n")
                total_files += file_count
        
        f.write(f"\n总文件数: {total_files}\n")
        f.write(f"目录结构: 4个统一的文件类型目录\n")
    
    print(f"处理摘要已保存到: {report_file}")

if __name__ == '__main__':
    try:
        main()
        
        # 生成摘要报告
        target_base_dir = '/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/merged_data'
        if os.path.exists(target_base_dir):
            create_summary_report(target_base_dir)
            
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
