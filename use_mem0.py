import asyncio
from copy import deepcopy
import datetime
import json
import os
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from dotenv import load_dotenv

from mem0.memory.main import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

import logging

def mute_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

# 静音日志
for logger_name in ['httpx', 'mem0.memory.main', 'openai._base_client']:
    mute_logger(logger_name)
load_dotenv()

@dataclass
class Mem0Config:
    """Mem0配置类 - 集中管理所有参数"""
    # === 可调整的核心参数 ===
    max_workers: int = 40
    search_limit: int = 100
    enable_graph: bool = False
    
    # === 文件路径参数 ===
    input_file: str = '/mnt/zh/dataset/copilot/longmemeval_s.json'
    output_dir: str = '/mnt/zh/outputs/copilot_rag/'
    output_file_postfix: str = '_async'
    
    # === 模型参数 ===
    embedding_dims: int = 1536
    embedding_model: str = 'text-embedding-3-small'
    llm_model: str = 'gpt-4o-mini'
    
    # === 数据库参数 ===
    redis_url: str = 'redis://localhost:6379'
    neo4j_url: str = 'bolt://localhost:7688'
    neo4j_username: str = 'neo4j'
    neo4j_password: str = '12345678'
    
    # === Azure参数 ===
    azure_api_key: str = 'EMPTY'
    azure_endpoint: str = 'https://westus2.papyrus.binginternal.com'
    papyrus_model_name: str = 'GPT4oMini-Batch'
    papyrus_quota_id: str = 'DeepSeekAMDAds'
    papyrus_timeout_ms: str = '1200000'
    
    # === 处理参数 ===
    max_retries: int = 3
    infer: bool = True
    
    @property
    def output_file(self) -> str:
        """生成输出文件路径"""
        mem0_type = "mem0plus" if self.enable_graph else "mem0"
        filename = f'longmemeval_s_all_type_{mem0_type}_top{self.search_limit}{self.output_file_postfix}.jsonl'
        return os.path.join(self.output_dir, filename)
    
    def get_memory_config_dict(self) -> Dict:
        """生成Memory配置字典"""
        config = {
            'vector_store': {
                'provider': 'redis',
                'config': {
                    'collection_name': None,  # 运行时动态设置
                    'redis_url': self.redis_url,
                    'embedding_model_dims': self.embedding_dims,
                }
            },
            'embedder': {
                'provider': 'azure_openai',
                'config': {
                    'embedding_dims': self.embedding_dims,
                    'model': self.embedding_model,
                }
            },
            'llm': {
                'provider': 'azure_openai',
                'config': {
                    'model': self.llm_model,
                }
            },
        }
        
        if self.enable_graph:
            config['graph_store'] = {
                'provider': 'neo4j',
                'config': {
                    'url': self.neo4j_url,
                    'username': self.neo4j_username,
                    'password': self.neo4j_password,
                    'llm': {
                        'provider': 'azure_openai_structured',
                        'config': {
                            'azure_kwargs': {
                                'api_key': self.azure_api_key,
                                'azure_endpoint': self.azure_endpoint,
                                'default_headers': {
                                    "Content-Type": "application/json",
                                    "papyrus-model-name": self.papyrus_model_name,
                                    "papyrus-quota-id": self.papyrus_quota_id,
                                    "papyrus-timeout-ms": self.papyrus_timeout_ms
                                }
                            }
                        }
                    }
                }
            }
        
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
        with open(self.config.input_file, 'r') as f:
            return json.load(f)
    
    def _get_processed_tasks(self) -> set:
        """获取已处理的任务"""
        if os.path.exists(self.config.output_file):
            try:
                res = pd.read_json(self.config.output_file, lines=True, orient="records")
                return set() if res.empty else set(res['question_id'])
            except:
                return set()
        return set()
    
    def run(self):
        """主执行函数"""
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取已处理的任务
        tasks_processed = self._get_processed_tasks()
        tasks_remain = [item for item in self.data if item['question_id'] not in tasks_processed]
        
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
        
        # 启动工作进程池
        with mp.Pool(self.config.max_workers) as pool:
            for result in tqdm(pool.imap(process_single_item, args_list), total=len(args_list), desc="处理进度"):
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
        question_id = item['question_id']
        start_time = time.time()
        
        # 创建Memory实例
        memory_config_dict = config.get_memory_config_dict()
        memory_config_dict['vector_store']['config']['collection_name'] = \
            f"mem0_collection_{question_id}_top{config.search_limit}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        memory_config = MemoryConfig(**memory_config_dict)
        memory = Memory(config=memory_config)
        
        # 重试机制处理单个项目
        for attempt in range(config.max_retries):
            memory.delete(memory_id=question_id)
            min_chat_num = min(config.search_limit, sum(len(s) for s in item['haystack_sessions']))
            
            # 添加会话数据
            _add_sessions_to_memory(memory, item, question_id, config.infer, worker_id)
            
            # 搜索相关记忆
            search_results = memory.search(query=item['question'], user_id=question_id, limit=config.search_limit)
            
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
                    'worker_id': worker_id
                })
                
                if config.enable_graph and 'relations' in search_results:
                    result['relations'] = search_results['relations']
                
                total_time = time.time() - start_time
                print(f"[Worker-{worker_id}] 问题 {question_id} 处理完成，总耗时 {total_time:.2f}s，检索率 {retrieval_rate:.4f}")
                return result
            else:
                print(f"[Worker-{worker_id}] 问题 {question_id} 搜索结果不足，重试 {attempt+1}/{config.max_retries}")
        
        print(f"[Worker-{worker_id}] 问题 {question_id} 处理失败，重试次数超过限制")
        return None
        
    except Exception as e:
        import traceback
        print(f"[Worker-{worker_id}] 处理问题 {item.get('question_id', 'unknown')} 时出错: {e}")
        traceback.print_exc()
        return None


def _add_sessions_to_memory(memory: Memory, item: Dict, question_id: str, infer: bool, worker_id: int):
    """添加会话到memory"""
    add_start = time.time()
    session_items = list(zip(item['haystack_sessions'], item['haystack_session_ids'], item['haystack_dates']))
    
    for haystack_session, haystack_session_id, haystack_date in tqdm(
        session_items, 
        desc=f'[Worker-{worker_id}] for Question {question_id}', 
        position=worker_id
    ):
        memory.add(
            haystack_session,
            user_id=question_id,
            metadata={'haystack_session_id': haystack_session_id, 'haystack_date': haystack_date},
            infer=infer,
        )
    
    add_time = time.time() - add_start
    print(f"[Worker-{worker_id}] 问题 {question_id} 添加会话完成，耗时 {add_time:.2f}s")


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
    print(f'平均每个问题的 topk 数量: {results_df["topk_session_ids"].apply(len).mean():.2f}')
    
    if 'relations' in results_df.columns:
        relations_count = results_df['relations'].apply(lambda x: len(x) if x else 0).sum()
        print(f'总关系数量: {relations_count}')
    
    if 'worker_id' in results_df.columns:
        worker_stats = results_df['worker_id'].value_counts().sort_index()
        print(f'各进程处理数量: {worker_stats.to_dict()}')


def run_mem0_experiment(enable_graph: bool = False):
    """运行 mem0 实验的主入口函数"""
    config = Mem0Config(enable_graph=enable_graph, infer=enable_graph)
    
    processor = MultiprocessMem0Processor(config)
    processor.run()
    print_statistics(config.output_file)


def main():
    """主函数"""
    import sys
    
    enable_graph = len(sys.argv) > 1 and sys.argv[1] == '--enable-graph'
    
    if enable_graph:
        print("使用多进程 mem0+ (带图数据库)")
    else:
        print("使用多进程标准 mem0 (不带图数据库)")
    
    run_mem0_experiment(enable_graph=enable_graph)
    print("多进程处理完成!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()