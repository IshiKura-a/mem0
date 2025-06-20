from copy import deepcopy
import datetime
import json
import os
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from dotenv import load_dotenv
import yaml

from mem0.memory.main import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

import logging

def mute_logger(name: str, level: int = logging.WARNING):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

# 静音日志
for logger_name in ['httpx', 'mem0.memory.main', 'openai._base_client', 'mem0.memory.graph_memory']:
    mute_logger(logger_name)

from transformers import logging
logging.set_verbosity_error()

load_dotenv()

class Mem0Config:
    """Mem0配置类 - 从config.yaml文件读取配置"""
    
    # === 可调整的核心参数 ===
    max_workers: int
    search_limit: int
    enable_graph: bool
    
    # === 文件路径参数 ===
    input_file: str
    output_dir: str
    output_file_prefix: str
    output_file_postfix: str
    
    # === 处理参数 ===
    max_retries: int
    infer: bool
    id_key: str
    
    # === Memory配置 ===
    memory_config: Dict
    
    def __init__(self, config_path: str = 'config.yaml'):
        """从YAML文件初始化配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 设置所有属性
        self.max_workers = config_data['max_workers']
        self.search_limit = config_data['search_limit']
        self.enable_graph = config_data['enable_graph']
        
        self.input_file = config_data['input_file']
        self.output_dir = config_data['output_dir']
        self.output_file_prefix = config_data['output_file_prefix']
        self.output_file_postfix = config_data['output_file_postfix']
        
        self.max_retries = config_data['max_retries']
        self.infer = config_data['infer']
        self.id_key = config_data['id_key']
        
        self.memory_config = config_data['memory_config']
    
    @property
    def output_file(self) -> str:
        """生成输出文件路径"""
        mem0_type = "mem0plus" if self.enable_graph else "mem0"
        filename = f'{self.output_file_prefix}_{mem0_type}_top{self.search_limit}{self.output_file_postfix}.jsonl'
        return os.path.join(self.output_dir, filename)
    
    def get_memory_config_dict(self) -> Dict:
        """从YAML配置生成Memory配置字典"""
        config = deepcopy(self.memory_config)
        
        # 如果不启用图数据库，移除graph_store配置
        if not self.enable_graph and 'graph_store' in config:
            del config['graph_store']
        
        return config


class MultiprocessMem0Processor:
    """多进程版本的 Mem0 处理器"""
    
    def __init__(self, config: Mem0Config):
        self.config = config
        self.data = self._load_data()
        self.tot_processed = 0
        self.tot_failed = 0
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        if self.config.input_file.endswith('.jsonl'):
            return pd.read_json(self.config.input_file, lines=True, orient="records").to_dict(orient='records')
        elif self.config.input_file.endswith('.json'):
            with open(self.config.input_file, 'r') as f:
                return json.load(f)
        else:
            raise NotImplementedError(f"不支持的输入文件格式: {self.config.input_file}")
    
    def _get_processed_tasks(self) -> set:
        """获取已处理的任务"""
        if os.path.exists(self.config.output_file):
            try:
                res = pd.read_json(self.config.output_file, lines=True, orient="records")
                return set() if res.empty else set(res[self.config.id_key])
            except:
                return set()
        return set()
    
    def run(self):
        """主执行函数"""
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取已处理的任务
        tasks_processed = self._get_processed_tasks()
        tasks_remain = [item for item in self.data if item[self.config.id_key] not in tasks_processed]
        
        print(f"已处理任务: {len(tasks_processed)} / {len(self.data)}")
        print(f'正在处理文件: {self.config.output_file}')
        print(f'使用进程数: {self.config.max_workers}')
        print(f'剩余任务数: {len(tasks_remain)}')
        
        if not tasks_remain:
            print("所有任务已完成！")
            return
        
        # 准备多进程参数
        args_list = [(item, self.config, i % self.config.max_workers) for i, item in enumerate(tasks_remain)]
        
        self._execute_multiprocess(args_list)
    
    def _execute_multiprocess(self, args_list: List):
        """执行多进程处理"""
        start_time = time.time()
        
        # 设置多进程管理器和队列
        manager = mp.Manager()
        queue = manager.Queue()
        
        # 启动写入进程
        writer_process = mp.Process(target=writer_worker, args=(queue, self.config.output_file))
        writer_process.start()
        
        print(f'开始使用 {self.config.max_workers} 个进程处理 {len(args_list)} 个任务')
        
        # 根据输出文件前缀选择不同的处理函数
        if 'locomo' in self.config.output_file_prefix.lower():
            process_func = process_single_item_locomo
        elif 'longmemeval' in self.config.output_file_prefix.lower():
            process_func = process_single_item
        elif 'personamem' in self.config.output_file_prefix.lower():
            process_func = process_single_item_personamem
        else:
            raise NotImplementedError(f"不支持的数据集类型: {self.config.output_file_prefix}")
        
        # 启动工作进程池
        with mp.Pool(self.config.max_workers) as pool:
            for result in tqdm(pool.imap(process_func, args_list), total=len(args_list), desc="处理进度"):
                if result is not None:
                    queue.put(result)
                    self.tot_processed += 1
                else:
                    self.tot_failed += 1
        
        queue.put(None)
        writer_process.join()
        
        total_time = time.time() - start_time
        print("处理完成! 结果保存在:", self.config.output_file)
        print(f'总处理数量: {self.tot_processed}')
        print(f'失败数量: {self.tot_failed}')
        print(f'总耗时: {total_time:.2f}s')
        print(f'平均每个任务耗时: {total_time/len(args_list):.2f}s')


def process_single_item(args) -> Optional[Dict]:
    """多进程工作函数"""
    item, config, worker_id = args
    
    try:
        item_id = item[config.id_key]
        start_time = time.time()
        
        # 创建Memory实例
        memory_config_dict = config.get_memory_config_dict()
        memory_config_dict['vector_store']['config']['collection_name'] = \
            f"mem0_collection_longmemeval_{item_id}_top{config.search_limit}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        memory_config = MemoryConfig(**memory_config_dict)
        memory = Memory(config=memory_config)
        
        # 重试机制处理单个项目
        for attempt in range(config.max_retries):
            memory.delete(memory_id=item_id)
            min_chat_num = min(config.search_limit, sum(len(s) for s in item['haystack_sessions']))
            
            add_start = time.time()
            session_items = list(zip(item['haystack_sessions'], item['haystack_session_ids'], item['haystack_dates']))
            
            for haystack_session, haystack_session_id, haystack_date in tqdm(
                session_items,
                desc=f'[Worker-{worker_id}] for Question {item_id}',
                position=worker_id
            ):
                memory.add(
                    haystack_session,
                    user_id=item_id,
                    metadata={'haystack_session_id': haystack_session_id, 'haystack_date': haystack_date},
                    infer=config.infer,
                )
            
            add_time = time.time() - add_start
            print(f"[Worker-{worker_id}] 问题 {item_id} 添加会话完成，耗时 {add_time:.2f}s")
            
            # 搜索相关记忆
            search_results = memory.search(query=item['question'], user_id=item_id, limit=config.search_limit)
            
            # 检查搜索结果是否满足条件
            if len(search_results['results']) >= min_chat_num:
                # 计算检索率
                topk_session_ids = [result['metadata']['haystack_session_id'] for result in search_results['results']]
                answer_retrieved = [session_id in topk_session_ids for session_id in item['answer_session_ids']]
                retrieval_rate = sum(answer_retrieved) / len(item['answer_session_ids']) if item['answer_session_ids'] else 0
                
                # 准备返回结果
                result = item.copy()
                result.update({
                    'topk_session_ids': topk_session_ids,
                    'answer_retrived': answer_retrieved,
                    'retrieval_rate': retrieval_rate,
                    'search_results': search_results['results'],
                    'mem0_type': 'mem0+' if config.enable_graph else 'mem0',
                    'worker_id': worker_id,
                    'collection_name': memory_config_dict['vector_store']['config']['collection_name']
                })
                
                if config.enable_graph and 'relations' in search_results:
                    result['relations'] = search_results['relations']
                
                total_time = time.time() - start_time
                print(f"[Worker-{worker_id}] 问题 {item_id} 处理完成，总耗时 {total_time:.2f}s，检索率 {retrieval_rate:.4f}")
                return result
            else:
                print(f"[Worker-{worker_id}] 问题 {item_id} 搜索结果不足，重试 {attempt+1}/{config.max_retries}")
        
        print(f"[Worker-{worker_id}] 问题 {item_id} 处理失败，重试次数超过限制")
        return None
        
    except Exception as e:
        import traceback
        print(f"[Worker-{worker_id}] 处理问题 {item.get(config.id_key, 'unknown')} 时出错: {e}")
        traceback.print_exc()
        return None


def process_single_item_locomo(args) -> Optional[Dict]:
    """多进程工作函数"""
    item, config, worker_id = args
    
    try:
        item_id = item[config.id_key]
        start_time = time.time()
        
        # 创建Memory实例
        memory_config_dict = config.get_memory_config_dict()
        memory_config_dict['vector_store']['config']['collection_name'] = \
            f"mem0_collection_locomo_{item_id}_top{config.search_limit}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        memory_config = MemoryConfig(**memory_config_dict)
        memory = Memory(config=memory_config)
        
        # 重试机制处理单个项目
        for attempt in range(config.max_retries):
            memory.delete(memory_id=item_id)
            min_chat_num = min(
                config.search_limit,
                sum(len(s) for v in item['conversation'].values() if isinstance(v, list) for s in v)
            )
            
            for k,v in tqdm(item['conversation'].items(),
                            desc=f'[Worker-{worker_id}] for Question {item_id}',
                            position=worker_id):
                if isinstance(v, list):
                    session = [{
                        'role': chat['speaker'],
                        'name': f'{chat["speaker"]}_{chat["dia_id"]}',
                        'content': chat['text']
                    } for chat in v]
                    memory.add(
                        session,
                        user_id=item_id,
                        metadata={'session_id': k, 'session_date': item['conversation'][f'{k}_date_time']},
                        infer=config.infer,
                    )
            
            search_results = []
            relations = []
            for qa in item['qa']:
                result = memory.search(query=qa['question'], user_id=item_id, limit=config.search_limit)
                search_results.append(result['results'])
                relations.append(result.get('relations', []))
            
            # 检查搜索结果是否满足条件
            if all([len(results) >= min_chat_num for results in search_results]):
                # 准备基础返回结果
                result = item.copy()
                base_result = {
                    'search_results': search_results,
                    'relations': relations,
                    'mem0_type': 'mem0+' if config.enable_graph else 'mem0',
                    'worker_id': worker_id,
                    'collection_name': memory_config_dict['vector_store']['config']['collection_name']
                }
                
                if config.infer:
                    # 推理模式：设置检索率为-1
                    base_result['retrieval_rate'] = -1
                else:
                    # 非推理模式：计算具体的检索率和相关数据
                    topk_dia_ids = [[_['actor_id'].split('_')[-1] for _ in results] for results in search_results]
                    answer_retrieved = [
                        [e in topk_dia_ids_per_qa for e in qa['evidence']]
                        for qa, topk_dia_ids_per_qa in zip(item['qa'], topk_dia_ids)
                    ]
                    retrieval_rate = sum([sum(_) for _ in answer_retrieved]) / sum([len(_) for _ in answer_retrieved])
                    
                    base_result.update({
                        'topk_dia_ids': topk_dia_ids,
                        'answer_retrived': answer_retrieved,
                        'retrieval_rate': retrieval_rate
                    })
                
                result.update(base_result)
                
                total_time = time.time() - start_time
                retrieval_info = f"，检索率 {base_result['retrieval_rate']:.4f}" if not config.infer else ""
                print(f"[Worker-{worker_id}] 问题 {item_id} 处理完成，总耗时 {total_time:.2f}s{retrieval_info}")
                return result
            else:
                print(f"[Worker-{worker_id}] 问题 {item_id} 搜索结果不足，重试 {attempt+1}/{config.max_retries}")
        
        print(f"[Worker-{worker_id}] 问题 {item_id} 处理失败，重试次数超过限制")
        return None
        
    except Exception as e:
        import traceback
        print(f"[Worker-{worker_id}] 处理问题 {item.get(config.id_key, 'unknown')} 时出错: {e}")
        traceback.print_exc()
        return None


def process_single_item_personamem(args) -> Optional[Dict]:
    """多进程工作函数"""
    item, config, worker_id = args
    
    try:
        item_id = item[config.id_key]
        start_time = time.time()
        
        # 创建Memory实例
        memory_config_dict = config.get_memory_config_dict()
        memory_config_dict['vector_store']['config']['collection_name'] = \
            f"mem0_collection_personamem_{item_id}_top{config.search_limit}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        memory_config = MemoryConfig(**memory_config_dict)
        memory = Memory(config=memory_config)
        
        # 重试机制处理单个项目
        for attempt in range(config.max_retries):
            memory.delete(memory_id=item_id)
            min_chat_num = min(
                config.search_limit, len(item['contexts']) - 1
            )
            
            memory.add(
                item['contexts'][1:],
                user_id=item_id,
                metadata={},
                infer=config.infer,
            )
            
            search_results = []
            relations = []
            for qa in item['questions']:
                result = memory.search(query=qa['user_question_or_message'], user_id=item_id, limit=config.search_limit)
                search_results.append(result['results'])
                relations.append(result.get('relations', []))
            
            # 检查搜索结果是否满足条件
            if all([len(results) >= min_chat_num for results in search_results]):
                # 准备基础返回结果
                result = item.copy()
                result.update({
                    'search_results': search_results,
                    'relations': relations,
                    'mem0_type': 'mem0+' if config.enable_graph else 'mem0',
                    'worker_id': worker_id,
                    'retrieval_rate': -1,
                    'collection_name': memory_config_dict['vector_store']['config']['collection_name']
                })
                
                total_time = time.time() - start_time
                retrieval_info = f"，检索率 {result['retrieval_rate']:.4f}" if not config.infer else ""
                print(f"[Worker-{worker_id}] 问题 {item_id} 处理完成，总耗时 {total_time:.2f}s{retrieval_info}")
                return result
            else:
                print(f"[Worker-{worker_id}] 问题 {item_id} 搜索结果不足，重试 {attempt+1}/{config.max_retries}")
        
        print(f"[Worker-{worker_id}] 问题 {item_id} 处理失败，重试次数超过限制")
        return None
        
    except Exception as e:
        import traceback
        print(f"[Worker-{worker_id}] 处理问题 {item.get(config.id_key, 'unknown')} 时出错: {e}")
        traceback.print_exc()
        return None


def writer_worker(queue: mp.Queue, output_file: str):
    """独立的写入进程"""
    with open(output_file, "a+") as f:
        while True:
            result = queue.get()
            if result is None:  # 停止信号
                break
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_statistics(output_file: str):
    """打印统计信息"""
    if not os.path.exists(output_file):
        print("输出文件不存在")
        return
        
    results_df = pd.read_json(output_file, lines=True, orient="records")
    
    overall_retrieval_rate = results_df['retrieval_rate'].mean()
    mem0_type = results_df['mem0_type'].iloc[0] if 'mem0_type' in results_df.columns else 'unknown'
    
    print(f'使用的版本: {mem0_type}')
    print(f'总体检索率: {overall_retrieval_rate:.4f}')
    print(f'处理的问题总数: {len(results_df)}')
    if 'topk_session_ids' in results_df.columns:
        print(f'平均每个问题的 topk 数量: {results_df["topk_session_ids"].apply(len).mean():.2f}')
    elif 'topk_dia_ids' in results_df.columns:
        print(f'平均每个问题的 topk 对话数量: {results_df["topk_dia_ids"].apply(lambda x: np.mean([len(i) for i in x])).mean():.2f}')
    
    if 'relations' in results_df.columns:
        relations_count = results_df['relations'].apply(lambda x: len(x) if x else 0).sum()
        print(f'总关系数量: {relations_count}')
    
    if 'worker_id' in results_df.columns:
        worker_stats = results_df['worker_id'].value_counts().sort_index()
        print(f'各进程处理数量: {worker_stats.to_dict()}')


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mem0 多进程处理器')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='配置文件路径 (默认: config/config.yaml)')
    
    args = parser.parse_args()
    
    # 从配置文件读取设置
    config = Mem0Config(config_path=args.config)
    
    if config.enable_graph:
        print("使用多进程 mem0+ (带图数据库)")
    else:
        print("使用多进程标准 mem0 (不带图数据库)")
    
    print(f"使用配置文件: {args.config}")
    print(f"推理模式: {config.infer}")
    
    # 直接执行处理流程
    processor = MultiprocessMem0Processor(config)
    processor.run()
    print_statistics(config.output_file)
    print("多进程处理完成!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()