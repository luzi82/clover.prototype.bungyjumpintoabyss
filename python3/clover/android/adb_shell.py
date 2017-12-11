import subprocess

class AdbShell:

    def __init__(self, exec_path):
        self.exec_path = exec_path
        self.proc = None
    
    def connect(self):
        self.proc = subprocess.Popen(
            [self.exec_path,'shell'],
            stdin =subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def release(self):
        if self.proc is None: return
        self.proc.stdin.write(b'exit\n')
        self.proc.stdin.flush()
        self.proc.wait()
        self.proc = None
    
    def input_tap(self,x,y):
        cmd = 'input tap {} {}\n'.format(x,y)
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        #self.proc.communicate(input=cmd.encode('ascii'),timeout=1)
