from collections import OrderedDict
from node import Node
from utils import print_fn, _repr_job_preempt, _repr_job_done, large_job_pruning
from job_history import JobHistory

class Cluster:
    def __init__(self, node_list=None, num_nodes=None, num_gpus=20,
                 num_cpus=20, pattern=1, period=124, job_list=None,
                 random_seed=0, num_spare_node=None,
                 export_cluster_util=False):
        if node_list is not None:
            node_list = node_list
        elif num_nodes is not None:
            node_list = [Node(id=i) for i in range(num_nodes)]
        else:
            node_list = [Node(id=0, num_gpus=num_gpus, num_cpus=num_cpus)]

        temp_node_dict = dict()
        self.num_gpus, self.num_cpus = 0, 0
        for node in node_list:
            self.num_gpus += node.num_gpus
            self.num_cpus += node.num_cpus
            temp_node_dict[node.id] = node
        self.node_dict = OrderedDict(sorted(temp_node_dict.items(),
                                            key=lambda t: t[1].id))

        self.cur_time = 0
        self.svc = {'num_gpu': 0, 'num_cpu': 0} 
        self.svc_former_ratio = 0

        # 作业列表处理
        self.job_full_list = large_job_pruning(job_list, self.num_gpus, self.num_cpus)
        self.job_full_list.sort(key=lambda j: -j['submit_time'])
        self.job_list = []
        
        # 【性能优化】手动维护运行列表
        self.job_runn_list = [] 
        
        self.retrieve_job_from_full_list()  
        self.job_history = JobHistory()
        self.pattern = pattern
        self.period = period
        self.num_spare_node = num_spare_node
        self.spare_node_id = []
        
        if num_spare_node is not None:
            for i in range(num_spare_node):
                spare_node_index = random_seed % len(node_list)
                spare_node_id = node_list[spare_node_index].id
                while spare_node_id in self.spare_node_id:
                    random_seed += 29741 
                    spare_node_index = random_seed % len(node_list)
                    spare_node_id = node_list[spare_node_index].id
                self.spare_node_id.append(spare_node_id) 
                random_seed += 29741 

        self.export_cluster_util = export_cluster_util
        self.cluster_time = []
        self.cluster_cpu = []
        self.cluster_gpu = []
        self.idle_cluster_counter = 0

    def retrieve_job_from_full_list(self):
        while len(self.job_full_list) > 0:
            job = self.job_full_list[-1]
            if job['submit_time'] <= self.cur_time:
                job = self.job_full_list.pop()
                self.job_list.append(job)
            else:
                return 0

    def sorted_node_list(self):
        node_list = list(self.node_dict.values())
        node_list.sort(key=lambda n: n.id)
        return node_list

    def tic_job(self, delta=1):
        self.cur_time += delta
        if self.export_cluster_util and self.cur_time % 10000 == 0:
            self.record_cluster_util()
        self.retrieve_job_from_full_list() 
        
        job_runn_list = self.job_runn_list 
        
        if len(job_runn_list) > 0:
            # 必须复制一份遍历，因为循环中可能删除元素
            for job in job_runn_list[:]: 
                job['on_time'] += delta
                
                if job['on_time'] >= job['duration']:
                    over_tic_time = job['on_time'] - job['duration'] 
                    job['on_time'] -= over_tic_time
                    job['done'] = 1

                    host_node_id = job['node']
                    host_node = self.node_dict.get(host_node_id)
                    suc = host_node.release_job(job=job)
                    assert suc

                    job['jct'] = self.cur_time - over_tic_time - job['submit_time'] 
                    self.job_history.add_done_job(job)
                    
                    # 从运行列表移除
                    if job in self.job_runn_list:
                        self.job_runn_list.remove(job)

            return self.cur_time 

        elif len(self.job_list) > 0: 
            self.idle_cluster_counter += 1
            return self.cur_time 

        elif len(self.job_full_list) > 0:
            wake_time = self.job_full_list[-1]['submit_time'] - delta 
            if self.cur_time <= wake_time:
                self.cur_time = wake_time
            else:
                pass 
            return self.cur_time 

        else: 
            return -1 

    def tic_svc(self, cur_time):
        self.cur_time = cur_time
        cap_ratio = self.get_cap_ratio(cur_time)
        svc_ratio = 1 - cap_ratio
        if self.svc_former_ratio != svc_ratio:
            self.svc_former_ratio = svc_ratio
            for node in self.node_list:
                if node.id in self.spare_node_id: 
                    continue
                node.set_svc_res_by_ratio(ratio=svc_ratio)

    def get_cap_ratio(self, time, pattern=None, period=None):
        return 1

    def record_cluster_util(self):
        self.cluster_time.append(self.cur_time)
        self.cluster_cpu.append(self.job_cpus)
        self.cluster_gpu.append(self.job_gpus)
    
    @property
    def node_list(self):
        return list(self.node_dict.values())

    @property
    def cur_rsrc(self):
        return [self.cur_gpus, self.cur_cpus]

    @property
    def cur_gpus(self):
        return self.num_gpus - self.svc_gpus

    @property
    def cur_cpus(self):
        return self.num_cpus - self.svc_cpus

    @property
    def svc_gpus(self):
        return sum([n.svc_gpus for n in self.node_list])

    @property
    def svc_cpus(self):
        return sum([n.svc_cpus for n in self.node_list])

    @property
    def idl_gpus(self):
        """当前空闲 GPU 总数"""
        return sum([n.idl_gpus for n in self.node_list])

    @property
    def idl_cpus(self):
        """当前空闲 CPU 总数"""
        return sum([n.idl_cpus for n in self.node_list])

    @property
    def job_gpus(self):
        """当前作业占用的 GPU 总数"""
        return sum([n.job_gpus for n in self.node_list])

    @property
    def job_cpus(self):
        """当前作业占用的 CPU 总数"""
        return sum([n.job_cpus for n in self.node_list])
