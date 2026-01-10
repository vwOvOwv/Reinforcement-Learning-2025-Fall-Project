import atexit
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import resource_tracker
import _pickle as cPickle
import time
import random

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 2048
        self.shared_model_list = ShareableList([b' ' * metadata_size] * capacity + [self.n], name = name)
        
        atexit.register(self.cleanup)
        
    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity
        if self.model_list[n] is not None:
            try:
                old_shm = self.model_list[n]['memory']
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[Server Warning] Failed to unlink old memory: {e}")
        
        data = cPickle.dumps(state_dict) # model parameters serialized to bytes
        memory = SharedMemory(create = True, size = len(data))
        memory.buf[:] = data[:]
        # print('Created model', self.n, 'in shared memory', memory.name)
        
        client_metadata = metadata.copy()
        client_metadata['_addr'] = memory.name
        client_metadata['id'] = self.n

        server_record = {
            'memory': memory,
            'id': self.n
        }
        self.model_list[n] = server_record

        try:
            self.shared_model_list[n] = cPickle.dumps(client_metadata)
            self.n += 1
            self.shared_model_list[-1] = self.n
        except Exception as e:
            print(f"[Server Error] Failed to update shared list: {e}")
            memory.close()
            memory.unlink()

    def cleanup(self):
        print("[Server] Cleaning up shared memory...")
        
        # 1. 清理具体的模型内存块
        for item in self.model_list:
            if item is not None and 'memory' in item:
                try:
                    name = item['memory'].name
                    # 尝试从 tracker 注销，防止报警
                    resource_tracker.unregister(name, 'shared_memory')
                    item['memory'].close()
                    item['memory'].unlink()
                except Exception:
                    pass
        
        # 2. 清理共享列表本身 (正是你报错的那个 /model-pool-xxx)
        if hasattr(self, 'shared_model_list'):
            try:
                name = self.shared_model_list.shm.name
                # 【关键修复】告诉 resource_tracker 不要管这个文件了，我自己删
                resource_tracker.unregister(name, 'shared_memory')
                
                self.shared_model_list.shm.close()
                self.shared_model_list.shm.unlink()
            except FileNotFoundError:
                pass # 已经被删了，无所谓
            except KeyError:
                pass # 有时候 tracker 里可能没记录，忽略
            except Exception as e:
                print(f"[Cleanup Error] {e}")
                
        print("[Server] Cleanup done.")

class ModelPoolClient:
    
    def __init__(self, name):
        while True:
            try:
                self.shared_model_list = ShareableList(name = name)
                n = self.shared_model_list[-1]
                break
            except:
                time.sleep(0.1)
        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._update_model_list()
    
    def _update_model_list(self):
        try:
            n = self.shared_model_list[-1]
        except (ValueError, IndexError):
            return
        # n = self.shared_model_list[-1]
        # if n > self.n:
        #     # new models available, update local list
        #     for i in range(max(self.n, n - self.capacity), n):
        #         self.model_list[i % self.capacity] = cPickle.loads(self.shared_model_list[i % self.capacity])
        #     self.n = n
        if n > self.n:
            # new models available, update local list
            for i in range(max(self.n, n - self.capacity), n):
                idx = i % self.capacity
                try:
                    raw_data = self.shared_model_list[idx]
                    self.model_list[idx] = cPickle.loads(raw_data)
                except (EOFError, cPickle.UnpicklingError, IndexError):
                    self.model_list[idx] = None
                except Exception:
                    self.model_list[idx] = None
            self.n = n
    
    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend(self.model_list[self.n % self.capacity :])
        model_list.extend(self.model_list[: self.n % self.capacity])
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(0.1)
            self._update_model_list()

        for i in range(self.capacity):
            idx = (self.n - 1 - i) % self.capacity
            meta = self.model_list[idx]
            if meta is not None:
                return meta
            
        return None
    
    def get_random_model(self):
        self._update_model_list()
        if self.n == 0:
            return None
        min_id = max(0, self.n - self.capacity)
        max_id = self.n - 1
        rand_id = random.randint(min_id, max_id)
        idx = rand_id % self.capacity
        
        meta = self.model_list[idx]
        return meta
    
    def load_model(self, metadata):
        if metadata is None: 
            return None
        self._update_model_list()
        n = metadata['id']
        if n < self.n - self.capacity:
            return None
        # memory = SharedMemory(name = metadata['_addr'])
        # state_dict = cPickle.loads(memory.buf)
        memory = None
        try:
            memory = SharedMemory(name = metadata['_addr'])
            state_dict = cPickle.loads(memory.buf)
            return state_dict
        except FileNotFoundError:
            print(f"[Client] Warning: Model {n} memory not found (overwritten?). Skip.")
            return None
        except Exception as e:
            print(f"[Client] Error loading model {n}: {e}")
            return None
        finally:
            if memory is not None:
                memory.close()