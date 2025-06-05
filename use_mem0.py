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


mute_logger('httpx')
mute_logger('mem0.memory.main')
mute_logger('openai._base_client')
load_dotenv()

@dataclass
class Mem0Config:
    """Configuration class for Mem0 processing"""
    input_file: str
    output_dir: str
    output_filename: str
    max_workers: int
    embedding_dims: int
    search_limit: int
    enable_graph: bool = False
    infer: bool = True
    
    # Derived properties
    output_file: str = ""
    
    def __post_init__(self):
        """Set derived properties after initialization"""
        self.output_file = os.path.join(self.output_dir, self.output_filename)


class MultiprocessMem0Processor:
    """多进程版本的 Mem0 处理器"""
    
    def __init__(self, config: Mem0Config):
        self.config = config
        
        # 创建内存配置字典
        self.memory_config_dict = {
            'vector_store': {
                'provider': 'redis',
                'config': {
                    'collection_name': None,
                    'redis_url': 'redis://localhost:6379',
                    'embedding_model_dims': config.embedding_dims,
                }
            },
            'embedder': {
                'provider': 'azure_openai',
                'config': {
                    'embedding_dims': config.embedding_dims,
                    'model': 'text-embedding-3-small',
                }
            },
            'llm': {
                'provider': 'azure_openai',
                'config': {
                    'model': 'gpt-4o-mini',
                }
            },
        }
        
        # 如果启用图数据库，添加图存储配置
        if config.enable_graph:
            self.memory_config_dict['graph_store'] = {
                'provider': 'neo4j',
                'config': {
                    'url': 'bolt://localhost:7688',
                    'username': 'neo4j',
                    'password': '12345678',
                    'llm': {
                        'provider': 'azure_openai_structured',
                        'config': {
                            'azure_kwargs': {
                                'api_key': 'EMPTY',
                                'azure_endpoint': 'https://westus2.papyrus.binginternal.com',
                                'default_headers': {
                                    "Content-Type": "application/json",
                                    "papyrus-model-name": 'GPT4oMini-Batch',
                                    "papyrus-quota-id": "DeepSeekAMDAds",
                                    "papyrus-timeout-ms": "1200000"
                                }
                            }
                        }
                    }
                }
            }
        
        # 加载数据
        self.data = self._load_data()
        
        # 统计
        self.tot_processed = 0
        self.tot_failed = 0
    
    def _load_data(self) -> List[Dict]:
        """加载数据"""
        with open(self.config.input_file, 'r') as f:
            return json.load(f)
            # data = json.load(f)
            # data = [item for item in data if item['question_id'] == '031748ae']
            # return data
    
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
        print(f"已处理任务: {len(tasks_processed)} / {len(self.data)}")
        
        # 过滤未处理的任务
        tasks_remain = [item for item in self.data if item['question_id'] not in tasks_processed]
        
        print(f'正在处理文件: {self.config.output_file}')
        print(f'使用进程数: {self.config.max_workers}')
        print(f'剩余任务数: {len(tasks_remain)}')
        
        # 为测试限制处理数量
        tasks_remain = tasks_remain
        
        # 准备参数列表
        args_list = [
            (item, self.memory_config_dict, self.config.search_limit, self.config.enable_graph, self.config.infer, i % self.config.max_workers)
            for i, item in enumerate(tasks_remain)
        ]
        
        print(f"将使用 {self.config.max_workers} 个进程处理 {len(tasks_remain)} 个任务")
        
        # 使用多进程处理 - 参考 LLaMA-Factory 实现
        start_time = time.time()
        
        # 设置多进程管理器和队列
        manager = mp.Manager()
        queue = manager.Queue()
        
        # 启动写入进程
        writer_process = mp.Process(target=writer_worker, args=(queue, self.config.output_file))
        writer_process.start()
        
        # 启动工作进程池
        print(f'开始使用 {self.config.max_workers} 个进程处理 {"/".join(self.config.output_file.split("/")[-2:])}')
        
        with mp.Pool(self.config.max_workers) as pool:
            # 使用 partial 函数创建工作函数
            worker_func = partial(process_single_item)
            
            # 使用 imap 处理任务并显示进度
            for result in tqdm(
                pool.imap(worker_func, args_list),
                total=len(args_list),
                desc="处理进度"
            ):
                if result is not None:
                    queue.put(result)
                    self.tot_processed += 1
                else:
                    self.tot_failed += 1
        
        # 发送停止信号给写入进程
        queue.put(None)
        writer_process.join()
        
        total_time = time.time() - start_time
        
        # 打印统计
        print("处理完成! 结果保存在:", self.config.output_file)
        print(f'总处理数量: {self.tot_processed}')
        print(f'失败数量: {self.tot_failed}')
        print(f'总耗时: {total_time:.2f}s')
        print(f'平均每个任务耗时: {total_time/len(tasks_remain):.2f}s')
        

def run_mem0_experiment_multiprocess(enable_graph: bool = False):
    """运行多进程版本的 mem0 实验"""
    output_suffix = "mem0plus" if enable_graph else "mem0"
    
    topk = 100
    config = Mem0Config(
        input_file='/mnt/zh/dataset/copilot/longmemeval_s.json',
        output_dir='/mnt/zh/outputs/copilot_rag/',
        output_filename=f'longmemeval_s_all_type_{output_suffix}_top{topk}_async_try4.jsonl',
        max_workers=20,  # 可以根据CPU核心数调整
        embedding_dims=1536,
        search_limit=topk,
        enable_graph=enable_graph,
        infer=enable_graph
    )
    
    processor = MultiprocessMem0Processor(config)
    processor.run()
    print_statistics(config.output_file)


def create_memory_instance(config_dict: Dict) -> Memory:
    """创建内存实例的函数，用于多进程环境"""
    memory_config = MemoryConfig(**config_dict)
    return Memory(config=memory_config)


def process_single_item(args) -> Optional[Dict]:
    """多进程工作函数，接收参数元组"""
    item, memory_config_dict, search_limit, enable_graph, infer, worker_id = args
    return process_single_item_worker(item, memory_config_dict, search_limit, enable_graph, infer, worker_id)

def process_single_item_worker(item: Dict, memory_config_dict: Dict, search_limit: int, enable_graph: bool, infer: bool, worker_id: int) -> Optional[Dict]:
    """异步处理单个项目的函数"""
    try:
        question_id = item['question_id']
        start_time = time.time()
        
        # 在每个进程中创建独立的内存实例
        config = deepcopy(memory_config_dict)
        config['vector_store']['config']['collection_name'] = f"mem0_collection_{question_id}_top{search_limit}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        memory = create_memory_instance(config)
        
        cnt = 0
        max_retries = 3        
        while cnt < max_retries:
            memory.delete(memory_id=question_id)
            min_chat_num = min(search_limit, sum(len(s) for s in item['haystack_sessions']))
            
            # 添加会话
            add_start = time.time()
            session_items = list(zip(
                item['haystack_sessions'],
                item['haystack_session_ids'],
                item['haystack_dates']
            ))
            
            for haystack_session, haystack_session_id, haystack_date in tqdm(session_items, desc=f'[Worker-{worker_id}] for Question {question_id}', position=worker_id):
                memory.add(
                    haystack_session,
                    user_id=question_id,
                    metadata={
                        'haystack_session_id': haystack_session_id,
                        'haystack_date': haystack_date
                    },
                    infer=infer,
                )
            add_time = time.time() - add_start
            
            print(f"[Worker-{worker_id}] 问题 {question_id} 添加会话完成，耗时 {add_time:.2f}s")
            
            # 搜索相关记忆
            search_start = time.time()
            search_results = memory.search(
                query=item['question'],
                user_id=question_id,
                limit=search_limit
            )
            
            # check results
            if len(search_results['results']) < min_chat_num:
                cnt += 1
                print(f"[Worker-{worker_id}] 问题 {question_id} 搜索结果不足，重试次数: {cnt}/{max_retries}")
            else:
                search_time = time.time() - search_start
                print(f"[Worker-{worker_id}] 问题 {question_id} 搜索完成，耗时 {search_time:.2f}s，找到 {len(search_results['results'])} 个结果")
                
                # 处理结果
                topk_session_ids = [result['metadata']['haystack_session_id'] for result in search_results['results']]
                answer_retrieved = [session_id in topk_session_ids for session_id in item['answer_session_ids']]
                retrieval_rate = sum(answer_retrieved) / len(item['answer_session_ids']) if item['answer_session_ids'] else 0
                
                # 准备结果
                result = item.copy()
                result.update({
                    'topk_session_ids': topk_session_ids,
                    'answer_retrived': answer_retrieved,
                    'retrieval_rate': retrieval_rate,
                    'search_results': search_results['results'],
                    'mem0_type': 'mem0+' if enable_graph else 'mem0',
                    'worker_id': worker_id
                })
                
                # 如果启用了图功能，添加关系信息
                if enable_graph and 'relations' in search_results:
                    result['relations'] = search_results['relations']

                break
        
        if cnt >= max_retries:
            print(f"[Worker-{worker_id}] 问题 {question_id} 处理失败，重试次数超过限制")
            return None
        else:
            total_time = time.time() - start_time
            print(f"[Worker-{worker_id}] 问题 {question_id} 处理完成，总耗时 {total_time:.2f}s，检索率 {retrieval_rate:.4f}")
            
            return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def writer_worker(queue: mp.Queue, output_file: str):
    """独立的写入进程，负责将结果写入文件"""
    with open(output_file, "a+") as f:
        while True:
            result = queue.get()
            if result is None:  # 停止信号
                break
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_statistics(output_file: str):
    """打印统计信息"""
    if os.path.exists(output_file):
        results_df = pd.read_json(output_file, lines=True, orient="records")
        
        if not results_df.empty and 'retrieval_rate' in results_df.columns:
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


def main():
    """主函数"""
    import sys
    
    # 解析命令行参数
    enable_graph = len(sys.argv) > 1 and sys.argv[1] == '--graph'
    
    if enable_graph:
        print("使用多进程 mem0+ (带图数据库)")
        run_mem0_experiment_multiprocess(enable_graph=True)
    else:
        print("使用多进程标准 mem0 (不带图数据库)")
        run_mem0_experiment_multiprocess(enable_graph=False)
    
    print("多进程处理完成!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()