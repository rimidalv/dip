import numpy as np


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


class Graph:
    def __init__(self, graph):
        self.graph=graph
        # print("graph", graph)

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                # print("sub_graphs",sub_graphs)
                while self.graph[v, :].any():
                    where = np.where(self.graph[v, :])
                    # print("where", where)
                    v = where[0][0]
                    # print("where_0", where_0)
                    # v= where_0[0]
                    # print("v",v)
                    sub_graphs[-1].append(v)
        return sub_graphs

