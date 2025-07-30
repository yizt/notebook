import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import struct

def worker(shm_name, lock):
    # 连接到共享内存
    shm = SharedMemory(name=shm_name)
    # 获取锁
    with lock:
        # 读取共享内存中的数据
        int_val = struct.unpack('i', shm.buf[0:4])[0]
        print(f"[子进程] 读取: int={int_val}")

        # 修改数据
        new_val = int_val + 1
        struct.pack_into('i', shm.buf, 0, new_val)
        print(f"[子进程] 修改: int={new_val}")

        # 关闭共享内存连接
        shm.close()

if __name__ == '__main__':
    # 创建共享内存（4字节，用于存储一个整数）
    shm = SharedMemory(name='share_memory_lock',create=True, size=4)
    # 初始化共享内存中的数据
    struct.pack_into('i', shm.buf, 0, 0)

    # 创建锁
    
    lock = multiprocessing.Lock()
    # 启动多个子进程
    processes = []
    for i in range(5):  # 启动 5 个子进程
        
        p = multiprocessing.Process(target=worker, args=(shm.name, lock))
        p.start()
        processes.append(p)

    # 等待所有子进程完成
    for p in processes:
        p.join()

    # 读取最终数据
    final_val = struct.unpack('i', shm.buf[0:4])[0]
    print(f"[主进程] 最终数据: int={final_val}")

    # 清理资源
    shm.close()
    shm.unlink()