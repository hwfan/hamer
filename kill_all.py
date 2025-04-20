import psutil
import os
import signal

def kill_other_move_gripper_processes():
    current_pid = os.getpid()  # 获取当前进程的PID
    killed = 0

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查是否是 move_gripper.py 进程 且 不是当前进程
            if (
                proc.info['cmdline']  # 确保有命令行参数
                and ('move_gripper.py' in ' '.join(proc.info['cmdline'])  # 匹配目标脚本
                     or
                     'move_gripper_old.py' in ' '.join(proc.info['cmdline'])  # 匹配目标脚本
                )
                and proc.info['pid'] != current_pid  # 排除自己
            ):
                print(f"杀死进程 PID={proc.info['pid']}, 命令行: {' '.join(proc.info['cmdline'])}")
                os.kill(proc.info['pid'], signal.SIGKILL)  # 发送终止信号
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    print(f"共杀死 {killed} 个 move_gripper.py 进程（排除自己）")

if __name__ == '__main__':
    kill_other_move_gripper_processes()
