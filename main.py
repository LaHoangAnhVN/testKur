import math
import random
import numpy as np

data_base = [[random.randint(1, 10), random.randint(-10, 0), random.randint(1, 10)] for i in range(100)]


def active_func(x):
    return (math.exp(2*x) - 1) / (math.exp(2*x) + 1)


def cost_func(actual, resolve):
    return (min(actual) - min(resolve))**2 + (max(actual) - max(resolve))**2


def loss_func(list_cost_func):
    return sum(list_cost_func)/len(list_cost_func)


def resolve_quadratic_equation(a: float, b: float, c: float):
    delta = b**2 - 4*a*c
    if delta < 0:
        return [0, 0]
    else:
        x1 = (-b - math.sqrt(delta))/(2*a)
        x2 = (-b + math.sqrt(delta))/(2*a)
        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if x1 <= x2:
            return [x1, x2]
        else:
            return [x2, x1]


def mean(massive):
    massive_mean = []
    for i in range(len(massive[0])):
        sum_column = [0, 0]
        for j in massive:
            sum_column[0] += j[i][0]
            sum_column[1] += j[i][1]
        massive_mean.append(np.array([sum_column[0]/len(massive), sum_column[1]/len(massive)]))
    return massive_mean


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

    def get_loss(self, input_data):
        expected_resolve = resolve_quadratic_equation(input_data[0], input_data[1], input_data[2])
        return [(expected_resolve[0] - self.get_output(input_data)[0]) ** 2, (expected_resolve[1] - self.get_output(input_data)[1])**2]

    def find_neural(self, id):
        for i in self.list_neural:
            for j in i:
                if j.id == id:
                    return j

    def find_layer(self, actual_neural: Neural):
        for i in range(len(self.list_neural)):
            if actual_neural in self.list_neural[i]:
                return i

    def dl_by_dx(self, input_data):

        expected_resolve = resolve_quadratic_equation(input_data[0], input_data[1], input_data[2])
        return [2 * (self.get_output(input_data)[0] - expected_resolve[0]), 2 * (self.get_output(input_data)[1] - expected_resolve[1])]

    def dti_by_dzi(self, neural: Neural):
        return neural.bias

    def dzi_by_dtk(self, left: Neural, right: Neural):
        layer = self.find_layer(left)
        for i in self.list_edges[layer]:
            if i.left == left.id and i.right == right.id:
                return i.weight

    def dz_by_dw(self, edge: Edge):
        return self.find_neural(edge.left).get_output()

    def dt_by_dk(self, neural: Neural):
        return neural.input_data

    def find_edge(self, left: int, right: int):
        left_neural = self.find_neural(left)
        layer = self.find_layer(left_neural)
        for i in self.list_edges[layer]:
            if i.left == left and i.right == right:
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

        # if last_neural.id == 7:
        #     check = False
        # else:
        #     check = True

        road = self.find_road(next_neural, last_neural)
        result = [0, 0]
        for i in road:
            list_edges = []
            actual = self.dl_by_dx(input_data)
            for j in range(len(i)):
                actual *= self.dti_by_dzi(i[j])
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in list_edges:
                actual *= self.dzi_by_dtk(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dz_by_dw(edge)
            result += actual
        return result
        # if check:
        #     return result[0]
        # else:
        #     return result[1]

    def dl_by_dk(self, input_data, actual_neural: Neural, last_neural: Neural):
        # check = True
        # if last_neural.id == 7:
        #     check = False
        road = self.find_road(actual_neural, last_neural)
        result = [0, 0]
        for i in road:
            list_edges = []
            actual = self.dl_by_dx(input_data)
            for j in range(1, len(i)):
                actual *= self.dti_by_dzi(i[j])
            for j in range(len(i)-1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in list_edges:
                actual *= self.dzi_by_dtk(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dt_by_dk(actual_neural)
            result += actual
        return result

    def training(self, data_training: []):
        expected_resolve = []
        for i in data_training:
            expected_resolve.append(resolve_quadratic_equation(i[0], i[1], i[2]))
        dl_by_dws = []
        dl_by_dks = []
        losses = []

        for i in range(len(data_training)):
            actual_output = self.get_output(data_training[i])
            actual_dk = []
            actual_dw = []
            for j in self.list_neural:
                for k in j:
                    x1 = self.dl_by_dk(data_training[i], k, self.find_neural(6))
                    x2 = self.dl_by_dk(data_training[i], k, self.find_neural(7))
                    actual_dk.append([x1, x2])
            for j in self.list_edges:
                for k in j:
                    x1 = self.dl_by_dw(data_training[i], k, self.find_neural(6))
                    x2 = self.dl_by_dw(data_training[i], k, self.find_neural(7))
                    actual_dw.append([x1, x2])

            dl_by_dks.append(actual_dk)
            dl_by_dws.append(actual_dw)

            actual_loss = (actual_output[0] - expected_resolve[i][0]) ** 2 + (
                    actual_output[1] - expected_resolve[i][1]) ** 2
            # print('x1: ', actual_output[0], 'x2: ', actual_output[1], 'X1: ', expected_resolve[i][0], 'X2: ', expected_resolve[i][1], 'Loss: ', actual_loss)
            losses.append(actual_loss)
        mean_dl_by_dws = mean(dl_by_dws)
        mean_dl_by_dks = mean(dl_by_dks)
        mean_losser = sum(losses) / len(losses)
        print(mean_losser)
        for i in self.list_neural:
            for j in i:
                j.bias -= self.learning_rate * mean_dl_by_dks[j.id - 1]
        k = 0
        for i in self.list_edges:
            for j in i:
                j.weight -= self.learning_rate * mean_dl_by_dws[k]
                k += 1
        return mean_losser

    def training_x1(self, data_training: []):
        expected_x1 = []
        for i in data_training:
            expected_x1.append(resolve_quadratic_equation(i)[0])

        dl1_by_dw1 = []
        dl1_by_dk1 = []
        losses_before = []

        for i in range(len(data_training)):
            losses_before.append(self.get_loss_x1(data_training[i]))
            actual_dl1_by_dw1 = []
            actual_dl1_by_dk1 = []

            for x in self.list_neural:
                for y in x:
                    actual_dl1_by_dk1.append(self.dl1_by_dk1(data_training[i], y))
            for x in self.list_edges:
                for y in x:
                    actual_dl1_by_dw1.append(self.dl1_by_dw1(data_training[i], y))
            dl1_by_dw1.append(actual_dl1_by_dw1)
            dl1_by_dk1.append(actual_dl1_by_dk1)
        mean_dl1_by_dw1 = mean_set(dl1_by_dw1)
        mean_dl1_by_dk1 = mean_set(dl1_by_dk1)
        mean_loss_before = sum(losses_before)/len(losses_before)

        for i in self.list_neural:
            for j in i:
                j.k -= self.learning_rate[0] * mean_dl1_by_dk1[j.id - 1]
        k = 0
        for i in self.list_edges:
            for j in i:
                j.weight -= self.learning_rate[0] * mean_dl1_by_dw1[k]
                k += 1
        losses_after = []
        for i in range(len(data_training)):
            losses_after.append(self.get_loss_x1(data_training[i]))
            # print('actual x1 = ', self.get_output(data_training[i])[0], 'X1 = ', resolve_quadratic_equation(data_training[i])[0])
        mean_loss_after = sum(losses_after)/len(losses_after)
        # print(mean_loss_before)
        # print(mean_loss_after)
        return abs(mean_loss_after - mean_loss_before)




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


a = [1.0, -3.0, 2.0]

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

net = Neural_Network(list_Neural, list_edge, np.array([0.1, 0.1]))
loss = net.training(data_base)

while loss > 0.05:
    loss = net.training(data_base)
    print(loss)
print('expected resolve: ', resolve_quadratic_equation(a[0], a[1], a[2]))
print('Output network: ', net.get_output(a))

