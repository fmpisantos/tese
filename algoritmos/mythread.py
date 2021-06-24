import threading


class myThread (threading.Thread):
    def __init__(self, func, objs, s, end):
        threading.Thread.__init__(self)
        self.func = func
        self.objs = objs
        self.s = s
        self.end = end

    def run(self):
        for idx in range(self.s,self.end):
            self.objs[idx] = self.func(self.objs[idx])
    
    def getObjs(self):
        return self.objs
        