import struct
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

def worker(shm_name):
    shm = SharedMemory(name=shm_name)
    # 读取数据（假设数据是 1个int + 1个float）
    int_val = struct.unpack('i', shm.buf[0:4])[0]
    float_val = struct.unpack('f', shm.buf[4:8])[0]
    print(f"[子进程] 读取: int={int_val}, float={float_val}")

    # 修改数据
    struct.pack_into('i', shm.buf, 0, 999)
    struct.pack_into('f', shm.buf, 4, 3.14)
    shm.close()

if __name__ == '__main__':
    # 创建共享内存（8字节：4字节int + 4字节float）
    # shm = SharedMemory(create=True, size=8)
    shm = SharedMemory(name='SharedMemory001')
    # 写入初始数据
    struct.pack_into('i', shm.buf, 0, 42)
    struct.pack_into('f', shm.buf, 4, 1.618)
    print("[主进程] 初始写入完成")

    # 启动子进程
    p = Process(target=worker, args=(shm.name,))
    p.start()
    p.join()

    # 读取子进程修改后的数据
    int_val = struct.unpack('i', shm.buf[0:4])[0]
    float_val = struct.unpack('f', shm.buf[4:8])[0]
    print(f"[主进程] 最终数据: int={int_val}, float={float_val}")

    # 清理资源
    shm.close()
    shm.unlink()