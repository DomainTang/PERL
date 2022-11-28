from heapq import heappush, heappop, nsmallest, heapify


class PQ(object):
    def __init__(self, data=None):   # 自动执行,self.Q从列表转化为堆
        if data is None:
            self.Q = []
        else:
            self.Q = data
            heapify(self.Q)     # 转化列表为堆

    def push(self, elem):
        heappush(self.Q, elem)  # 把elem 压入堆

    def pop(self):
        return heappop(self.Q)  # 从堆中弹出最小的元素

    def nsmallest(self, n):
        return nsmallest(n, self.Q)  # 返回堆中n个最小的元素

    def __len__(self):
        return len(self.Q)