import math
import random
import numpy as np

data_base = [[1, 3, 2], [1, 4, 3], [1, 5, 6], [1, 6, 5], [1, 7, 6], [1, 8, 7], [1, 9, 8], [2, 3, 1], [2, 5, 3],
             [2, 7, 5]]


def active_func(x):
    return (math.exp(2*x) - 1) / (math.exp(2*x) + 1)


def cost_func(actual, resolve):
    return (min(actual) - min(resolve))**2 + (max(actual) - max(resolve))**2


def loss_func(list_cost_func):
    return sum(list_cost_func)/len(list_cost_func)


def resolve_quadratic_equation(a: int, b: int, c: int):
    delta = b**2 - 4*a*c
    if delta < 0:
        return [0, 0]
    else:
        x1 = (-b - math.sqrt(delta))/(2*a)
        x2 = (-b + math.sqrt(delta))/(2*a)
        if x1 <= x2:
            return [x1, x2]
        else:
            return [x2, x1]


class Neural:
    def __init__(self, id, input_data, bias):
        self.id = id
        self.bias = bias
        self.input_data = input_data

    def get_output(self):
        return self.input_data * self.bias


class Edge:
    def __init__(self, left, right, weight):
        self.left = left
        self.right = right
        self.weight = weight


class Neural_Network:
    # list_neural = [[Neural(i, np.array([0, 0]), np.array([1, 1])) for i in range(1, 4)],
    #                [Neural(i, np.array([0, 0]), np.array([1, 1])) for i in range(4, 6)],
    #                [Neural(i, np.array([0, 0]), np.array([1, 1])) for i in range(6, 8)]]
    #
    # list_edges = [[Edge(1, 4, np.array([1, 1])), Edge(1, 5, np.array([1, 1])),
    #                Edge(2, 5, np.array([1, 1])),
    #                Edge(3, 4, np.array([1, 1])), Edge(3, 5, np.array([1, 1]))],
    #               [Edge(4, 6, np.array([1, 1])), Edge(5, 6, np.array([1, 1])),
    #                Edge(5, 7, np.array([1, 1]))]]

    def __init__(self, list_neural, list_edges, learning_rate):
        self.list_neural = list_neural.copy()
        self.list_edges = list_edges.copy()
        self.learning_rate = learning_rate
        # self.list_neural = [
        #     [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(1, 4)],
        #     [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(4, 6)],
        #     [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(6, 8)]]

        # self.list_edges = [[Edge(1, 4, np.array([random.random(), random.random()])),
        #                     Edge(1, 5, np.array([random.random(), random.random()])),
        #                     Edge(2, 5, np.array([random.random(), random.random()])),
        #                     Edge(3, 4, np.array([random.random(), random.random()])),
        #                     Edge(3, 5, np.array([random.random(), random.random()]))],
        #                    [Edge(4, 6, np.array([random.random(), random.random()])),
        #                     Edge(5, 6, np.array([random.random(), random.random()])),
        #                     Edge(5, 7, np.array([random.random(), random.random()]))]]
        #
        # self.learning_rate = 0.05

    # def __init__(self, input_parameters):
    #     self.expected_resolve = resolve_quadratic_equation(input_parameters[0], input_parameters[1], input_parameters[2])
    #     self.list_neural[0][0].input_data = np.array([input_parameters[0], input_parameters[0]])
    #     self.list_neural[0][1].input_data = np.array([input_parameters[1], input_parameters[1]])
    #     self.list_neural[0][2].input_data = np.array([input_parameters[2], input_parameters[2]])
    #
    #     for i in range(1, len(self.list_neural)):
    #         for j in self.list_neural[i]:
    #             edges_to_this_neural = []
    #             for k in self.list_edges[i - 1]:
    #                 if k.right == j.id:
    #                     edges_to_this_neural.append(k)
    #
    #             neural_to_this_neural = []
    #             for y in edges_to_this_neural:
    #                 for x in self.list_neural[i - 1]:
    #                     if x.id == y.left:
    #                         neural_to_this_neural.append(x)
    #
    #             for k in range(len(neural_to_this_neural)):
    #                 j.input_data += neural_to_this_neural[k].get_output() * edges_to_this_neural[k].weight

    def get_output(self, input_data):
        for i in self.list_neural:
            for j in i:
                j.input_data = 0
        self.list_neural[0][0].input_data = np.array([input_data[0], input_data[0]])
        self.list_neural[0][1].input_data = np.array([input_data[1], input_data[1]])
        self.list_neural[0][2].input_data = np.array([input_data[2], input_data[2]])

        for i in range(1, len(self.list_neural)):
            for j in self.list_neural[i]:
                edges_to_this_neural = []
                for k in self.list_edges[i - 1]:
                    if k.right == j.id:
                        edges_to_this_neural.append(k)

                neural_to_this_neural = []
                for y in edges_to_this_neural:
                    for x in self.list_neural[i - 1]:
                        if x.id == y.left:
                            neural_to_this_neural.append(x)

                for k in range(len(neural_to_this_neural)):
                    j.input_data += neural_to_this_neural[k].get_output() * edges_to_this_neural[k].weight

        return np.array([self.list_neural[-1][0].get_output()[0], self.list_neural[-1][1].get_output()[1]])

    def get_loss(self, input_data, check=True):
        expected_resolve = resolve_quadratic_equation(input_data[0], input_data[1], input_data[2])
        if check:
            return (expected_resolve[0] - self.get_output(input_data)[0][0])**2
        else:
            return (expected_resolve[1] - self.get_output(input_data)[1][1])**2

    def find_neural(self, id):
        for i in self.list_neural:
            for j in i:
                if j.id == id:
                    return j

    def find_layer(self, actual_neural: Neural):
        for i in range(len(self.list_neural)):
            if actual_neural in self.list_neural[i]:
                return i

    def dl_by_dx(self, input_data, check=True):
        expected_resolve = resolve_quadratic_equation(input_data[0], input_data[1], input_data[2])
        if check:
            return 2 * (self.get_output(input_data)[0] - expected_resolve[0])
        else:
            return 2 * (self.get_output(input_data)[1] - expected_resolve[1])

    def dti_by_dzi(self, neural: Neural, check=True):
        if check:
            return neural.bias[0]
        else:
            return neural.bias[1]

    def dzi_by_dtk(self, left: Neural, right: Neural, check=True):
        layer = self.find_layer(left)
        for i in self.list_edges[layer]:
            if i.left == left.id and i.right == right.id:
                if check:
                    return i.weight[0]
                else:
                    return i.weight[1]

    def dz_by_dw(self, edge: Edge, check=True):
        if check:
            return self.find_neural(edge.left).get_output()[0]
        else:
            return self.find_neural(edge.left).get_output()[1]

    def dt_by_dk(self, neural: Neural, check=True):
        if check:
            return neural.input_data[0]
        else:
            return neural.input_data[1]

    def find_edge(self, left: int, rigth: int):
        left_neural = self.find_neural(left)
        layer = self.find_layer(left_neural)
        for i in self.list_edges[layer]:
            if i.left == left and i.right == rigth:
                return i

    def find_neighbour_right(self, actual_neural:Neural):
        list_edges_neighbour = []
        list_neural_neighbour = []
        layer = self.find_layer(actual_neural)
        for i in self.list_edges[layer]:
            if i.left == actual_neural.id:
                list_edges_neighbour.append(i)
        for i in self.list_neural[layer + 1]:
            for j in list_edges_neighbour:
                if i.id == j.right:
                    list_neural_neighbour.append(i)
        return list_neural_neighbour

    def find_road(self, start_neural: Neural, end_neural: Neural):
        road = [[start_neural]]
        start_layer = self.find_layer(start_neural)
        for i in range(start_layer, 2):
            for j in range(len(road)):
                k = len(road[0])
                next_neural = self.find_neighbour_right(road[0][k-1])
                for x in next_neural:
                    a = road[0].copy()
                    a.append(x)
                    road.append(a)
                road.remove(road[0])
        result = []
        for i in road:
            if i[-1].id == end_neural.id:
                result.append(i)
        return result

    def dl_by_dw(self, input_data, edge: Edge, last_neural: Neural):
        next_neural = 0
        for i in self.list_neural:
            for j in i:
                if j.id == edge.right:
                    next_neural = j

        if last_neural.id == 7:
            check = False
        else:
            check = True

        road = self.find_road(next_neural, last_neural)
        result = [0, 0]
        for i in road:
            list_edges = []
            actual = self.dl_by_dx(input_data, check)
            for j in range(len(i)):
                actual *= self.dti_by_dzi(i[j], check)
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in list_edges:
                actual *= self.dzi_by_dtk(self.find_neural(j.left), self.find_neural(j.right), check)
            actual *= self.dz_by_dw(edge, check)
            result += actual

        if check:
            return result[0]
        else:
            return result[1]

    def dl_by_dk(self, input_data, actual_neural: Neural, last_neural: Neural) -> float:
        check = True
        if last_neural.id == 7:
            check = False
        road = self.find_road(actual_neural, last_neural)
        result = [0, 0]
        for i in road:
            list_edges = []
            actual = self.dl_by_dx(input_data, check)
            for j in range(1, len(i)):
                actual *= self.dti_by_dzi(i[j], check)
            for j in range(len(i)-1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in list_edges:
                actual *= self.dzi_by_dtk(self.find_neural(j.left), self.find_neural(j.right), check)
            actual *= self.dt_by_dk(actual_neural, check)
            result += actual
        if check:
            return result[0]
        else:
            return result[1]

    # def training(self, data_training: []):
    #     expected_resolve = []
    #     for i in data_training:
    #         expected_resolve.append(resolve_quadratic_equation(i[0], i[1], i[2]))
    #     last_neural = self.find_neural(6)
    #     for i in self.list_neural:
    #         for j in i:
    #             j.bias[0] -= self.learning_rate * self.dl_by_dk(j, last_neural)
    #     for i in self.list_edges:
    #         for j in i:
    #             j.weight[0] -= self.learning_rate * self.dl_by_dw(j, last_neural)
    #
    #     last_neural = self.find_neural(7)
    #     for i in self.list_neural:
    #         for j in i:
    #             j.bias[1] -= self.learning_rate * self.dl_by_dk(j, last_neural)
    #     for i in self.list_edges:
    #         for j in i:
    #             j.weight[1] -= self.learning_rate * self.dl_by_dw(j, last_neural)


a = [1.0, 3.0, 2.0]

list_Neural = [
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(1, 4)],
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(4, 6)],
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(6, 8)]]
list_edge = [[Edge(1, 4, np.array([random.random(), random.random()])),
              Edge(1, 5, np.array([random.random(), random.random()])),
              Edge(2, 5, np.array([random.random(), random.random()])),
              Edge(3, 4, np.array([random.random(), random.random()])),
              Edge(3, 5, np.array([random.random(), random.random()]))],
             [Edge(4, 6, np.array([random.random(), random.random()])),
              Edge(5, 6, np.array([random.random(), random.random()])),
              Edge(5, 7, np.array([random.random(), random.random()]))]]

net = Neural_Network(list_Neural, list_edge, [0.05, 0.05])
e = net.find_edge(1, 5)
last = net.find_neural(7)
