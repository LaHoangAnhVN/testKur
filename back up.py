import math
import random
import numpy as np
import matplotlib.pyplot as plt


def resolve_quadratic_equation(parameter: []):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
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
        return [min(x1, x2), max(x1, x2)]


def generate_data_base(N: int):
    result = []
    for i in range(N):
        done = False
        while True:
            if done:
                break
            a1 = random.randint(1, 10)
            b1 = random.randint(-10, 0)
            c1 = random.randint(1, 10)
            if b1**2 - 4*a1*c1 >= 0:
                result.append([a1, b1, c1])
                done = True
    for i in range(round(N/10)):
        done = False
        while True:
            if done:
                break
            a1 = random.randint(1, 10)
            b1 = random.randint(-10, 0)
            c1 = random.randint(1, 10)
            if b1 ** 2 - 4 * a1 * c1 < 0:
                result.append([a1, b1, c1])
                done = True
    return result


def mean_set(input_set: []):
    result = []
    for i in range(len(input_set[0])):
        sum_column = 0
        for j in range(len(input_set)):
            sum_column += input_set[j][i]
        result.append(sum_column/len(input_set))
    return result


class Neural:
    def __init__(self, id: int, k: []):
        self.id = id
        self.k = k

    def get_output(self, input_data: []):
        return [input_data[0] * self.k[0], input_data[1] * self.k[1]]


class Edge:
    def __init__(self, left: int, right: int, weight: []):
        self.left = left
        self.right = right
        self.weight = weight


class Neural_network:
    def __init__(self, list_neural: [], list_edges: [], learning_rate: []):
        self.list_neural = list_neural.copy()
        self.list_edges = list_edges.copy()
        self.learning_rate = learning_rate

    def get_output(self, input_para):
        for i in self.list_neural:
            for j in i:
                j.input_data = [0, 0]
        self.list_neural[0][0].input_data = [input_para[0], input_para[0]]
        self.list_neural[0][1].input_data = [input_para[1], input_para[1]]
        self.list_neural[0][2].input_data = [input_para[2], input_para[2]]

        for i in range(1, len(self.list_neural)):
            for j in self.list_neural[i]:
                list_edges_to_this_neural = []
                for k in self.list_edges[i - 1]:
                    if k.right == j.id:
                        list_edges_to_this_neural.append(k)
                list_neural_to_this_neural = []
                for k in self.list_neural[i - 1]:
                    for x in list_edges_to_this_neural:
                        if k.id == x.left:
                            list_neural_to_this_neural.append(k)

                for k in range(len(list_neural_to_this_neural)):
                    j.input_data[0] += list_neural_to_this_neural[k].get_output()[0] *\
                                       list_edges_to_this_neural[k].weight[0]
                    j.input_data[1] += list_neural_to_this_neural[k].get_output()[1] * list_edges_to_this_neural[k].weight[1]

        return [self.list_neural[-1][0].get_output()[0], self.list_neural[-1][1].get_output()[1]]

    def find_neural(self, id: int) -> Neural:
        for i in self.list_neural:
            for j in i:
                if j.id == id:
                    return j

    def find_layer(self, neural: Neural) -> int:
        for i in range(len(self.list_neural)):
            if neural in self.list_neural[i]:
                return i

    def find_edge(self, left: int, right: int):
        left_neural = self.find_neural(left)
        left_layer = self.find_layer(left_neural)
        for i in self.list_edges[left_layer]:
            if i.left == left and i.right == right:
                return i

    def find_right_neighbour_neural(self, actual_neural: Neural):
        list_right_neighbour_edges = []
        list_right_neighbour_neural = []
        actual_layer = self.find_layer(actual_neural)
        for i in self.list_edges[actual_layer]:
            if i.left == actual_neural.id:
                list_right_neighbour_edges.append(i)
        for i in self.list_neural[actual_layer + 1]:
            for j in list_right_neighbour_edges:
                if j.right == i.id:
                    list_right_neighbour_neural.append(i)
        return list_right_neighbour_neural

    def find_road(self, start_neural: Neural, last_neural: Neural):
        road = [[start_neural]]

        start_layer = self.find_layer(start_neural)
        last_layer = self.find_layer(last_neural)
        _actual_layer = start_layer
        while _actual_layer < last_layer:
            new_road = []
            for i in road:
                next_right = self.find_right_neighbour_neural(i[-1])
                for j in next_right:
                    i_copy = i.copy()
                    i_copy.append(j)
                    new_road.append(i_copy)
            road = new_road
            _actual_layer += 1
        result = []
        for i in road:
            if i[-1].id == last_neural.id:
                result.append(i)
        return result

    def get_loss_x1(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        if abs(actual_resolve[0] - expected_resolve[0]) < (actual_resolve[0] - expected_resolve[0])**2:
            return (actual_resolve[0] - expected_resolve[0])**2
        else:
            return abs(actual_resolve[0] - expected_resolve[0])

    def get_loss_x2(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        if abs(actual_resolve[1] - expected_resolve[1]) < (actual_resolve[1] - expected_resolve[1]) ** 2:
            return (actual_resolve[1] - expected_resolve[1]) ** 2
        else:
            return abs(actual_resolve[1] - expected_resolve[1])

    def dl1_by_dx1(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        if (actual_resolve[0] - expected_resolve[0])**2 > abs(actual_resolve[0] - expected_resolve[0]):
            return 2 * (actual_resolve[0] - expected_resolve[0])
        else:
            return (actual_resolve[0] - expected_resolve[0])/(np.sqrt((actual_resolve[0] - expected_resolve[0])**2))

    def dl2_by_dx2(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        if (actual_resolve[1] - expected_resolve[1]) ** 2 > abs(actual_resolve[1] - expected_resolve[1]):
            return 2 * (actual_resolve[1] - expected_resolve[1])
        else:
            return (actual_resolve[1] - expected_resolve[1]) / (np.sqrt((actual_resolve[1] - expected_resolve[1]) ** 2))

    def dti1_by_dzi1(self, neural: Neural):
        return neural.k[0]

    def dti2_by_dzi2(self, neural: Neural):
        return neural.k[1]

    def dzi1_by_dtk1(self, left: Neural, right: Neural):
        layer = self.find_layer(left)
        for i in self.list_edges[layer]:
            if i.left == left.id and i.right == right.id:
                return i.weight[0]

    def dzi2_by_dtk2(self, left: Neural, right: Neural):
        layer = self.find_layer(left)
        for i in self.list_edges[layer]:
            if i.left == left.id and i.right == right.id:
                return i.weight[1]

    def dz1_by_dw1(self, edge: Edge):
        return self.find_neural(edge.left).get_output()[0]

    def dz2_by_dw2(self, edge: Edge):
        return self.find_neural(edge.left).get_output()[1]

    def dt1_by_dk1(self, neural: Neural):
        return neural.input_data[0]

    def dt2_by_dk2(self, neural: Neural):
        return neural.input_data[1]

    def dl1_by_dw1(self, input_para: [], edge: Edge):
        next_neural = self.find_neural(edge.right)
        road = self.find_road(next_neural, self.find_neural(6))
        result = 0
        for i in road:
            actual = self.dl1_by_dx1(input_para)
            list_edges = []
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in i:
                actual *= self.dti1_by_dzi1(j)
            for j in list_edges:
                actual *= self.dzi1_by_dtk1(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dz1_by_dw1(edge)
            result += actual
        return result

    def dl2_by_dw2(self, input_para: [], edge: Edge):
        next_neural = self.find_neural(edge.right)
        road = self.find_road(next_neural, self.find_neural(7))
        result = 0
        for i in road:
            actual = self.dl2_by_dx2(input_para)
            list_edges = []
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j + 1].id))

            for j in i:
                actual *= self.dti2_by_dzi2(j)
            for j in list_edges:
                actual *= self.dzi2_by_dtk2(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dz2_by_dw2(edge)
            result += actual
        return result

    def dl1_by_dk1(self, input_para: [], neural: Neural):
        road = self.find_road(neural, self.find_neural(6))
        result = 0
        for i in road:
            list_edges = []
            actual = self.dl1_by_dx1(input_para)
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))

            for j in range(1, len(i)):
                actual *= self.dti1_by_dzi1(i[j])
            for j in list_edges:
                actual *= self.dzi1_by_dtk1(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dt1_by_dk1(neural)
            result += actual
        return result

    def dl2_by_dk2(self, input_para, neural: Neural):
        road = self.find_road(neural, self.find_neural(7))
        result = 0
        for i in road:
            list_edges = []
            actual = self.dl2_by_dx2(input_para)
            for j in range(len(i) - 1):
                list_edges.append(self.find_edge(i[j].id, i[j+1].id))
            for j in range(1, len(i)):
                actual *= self.dti2_by_dzi2(i[j])
            for j in list_edges:
                actual *= self.dzi2_by_dtk2(self.find_neural(j.left), self.find_neural(j.right))
            actual *= self.dt2_by_dk2(neural)
            result += actual
        return result

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
                j.k[0] -= self.learning_rate[0] * mean_dl1_by_dk1[j.id - 1]
        k = 0
        for i in self.list_edges:
            for j in i:
                j.weight[0] -= self.learning_rate[0] * mean_dl1_by_dw1[k]
                k += 1
        losses_after = []
        for i in range(len(data_training)):
            losses_after.append(self.get_loss_x1(data_training[i]))

        mean_loss_after = sum(losses_after)/len(losses_after)
        return mean_loss_before, mean_loss_after

    def training_x2(self, data_training: []):
        expected_x2 = []
        for i in data_training:
            expected_x2.append(resolve_quadratic_equation(i)[1])
        dl2_by_dw2 = []
        dl2_by_dk2 = []
        losses_before = []

        for i in range(len(data_training)):
            losses_before.append(self.get_loss_x2(data_training[i]))
            actual_dl2_by_dw2 = []
            actual_dl2_by_dk2 = []

            for x in self.list_neural:
                for y in x:
                    actual_dl2_by_dk2.append(self.dl2_by_dk2(data_training[i], y))
            for x in self.list_edges:
                for y in x:
                    actual_dl2_by_dw2.append(self.dl2_by_dw2(data_training[i], y))

            dl2_by_dw2.append(actual_dl2_by_dw2)
            dl2_by_dk2.append(actual_dl2_by_dk2)

        mean_dl2_by_dw2 = mean_set(dl2_by_dw2)
        mean_dl2_by_dk2 = mean_set(dl2_by_dk2)
        mean_loss_before = sum(losses_before)/len(losses_before)

        for i in self.list_neural:
            for j in i:
                j.k[1] -= self.learning_rate[1] * mean_dl2_by_dk2[j.id - 1]
        k = 0
        for i in self.list_edges:
            for j in i:
                j.weight[1] -= self.learning_rate[1] * mean_dl2_by_dw2[k]
                k += 1
        losses_after = []
        for i in range(len(data_training)):
            losses_after.append(self.get_loss_x2(data_training[i]))

        mean_loss_after = sum(losses_after)/len(losses_after)
        return mean_loss_before, mean_loss_after

    def progress_training(self, condition=None, k=None):
        data_training = np.loadtxt('data.txt', dtype=int)
        list_mean_loss_x1 = []
        list_mean_loss_x2 = []
        min_neural = []
        min_neural_x1 = []
        min_edge_x1 = []
        min_edge = []
        list_network_x1 = []
        min_edge = []
        list_network_x2 = []
        if k is None and condition is not None:
            while True:
                before_x1, after_x1 = self.training_x1(data_training)
                before_x2, after_x2 = self.training_x2(data_training)
                list_mean_loss_x1.append(after_x1)
                list_mean_loss_x2.append(after_x2)
                if after_x1 < before_x1:
                    for i in range(len(self.list_neural)):
                        for j in range(len(self.list_neural[i])):
                            min_neural[i][j].k[0] = self.list_neural[i][j].k[0]
                    for i in range(len(self.list_edges)):
                        for j in range(len(self.list_edges[i])):
                            min_edge[i][j].weight[0] = self.list_edges[i][j].weight[0]
                if after_x2 < before_x2:
                    for i in range(len(self.list_neural)):
                        for j in range(len(self.list_neural[i])):
                            min_neural[i][j].k[1] = self.list_neural[i][j].k[1]
                    for i in range(len(self.list_edges)):
                        for j in range(len(self.list_edges[i])):
                            min_edge[i][j].weight[1] = self.list_edges[i][j].weight[1]
                if abs(before_x2 - after_x2) < condition and abs(before_x1 - after_x1) < condition:
                    break
        else:
            for x in range(k):
                before_x1, after_x1 = self.training_x1(data_training)
                list_mean_loss_x1.append(after_x1)
                list_k1 = []
                for i in self.list_neural:
                    t1 = []
                    for j in i:

                        new_neural = Neural(j.id, [0, 0], j.k)
                        t1.append(new_neural)
                    list_k1.append(t1)
                min_neural_x1.append(list_k1)
                list_weight1 = []
                for i in self.list_edges:
                    t2 = []
                    for j in i:
                        new_edge = Edge(j.left, j.right, j.weight)
                        t2.append(new_edge)
                    list_weight1.append(t2)
                min_edge_x1.append(list_weight1)

                # if after_x1 < before_x1:
                #     for i in range(len(self.list_neural)):
                #         for j in range(len(self.list_neural[i])):
                #             min_neural[i][j].k[0] = self.list_neural[i][j].k[0]
                #     for i in range(len(self.list_edges)):
                #         for j in range(len(self.list_edges[i])):
                #             min_edge[i][j].weight[0] = self.list_edges[i][j].weight[0]
            for x in range(k):
                before_x2, after_x2 = self.training_x2(data_training)
                list_mean_loss_x2.append(after_x2)
                list_network_x2.append(self)

            a = [1, -12, 35]
            test_1 = Neural_network(min_neural_x1[10], min_edge_x1[10], self.learning_rate)
            test_2 = Neural_network(min_neural_x1[50], min_edge_x1[50], self.learning_rate)
            print(test_1.get_output(a))
            print(test_2.get_output(a))

            # min_x1 = min(list_mean_loss_x1)
            # min_index_x1 = list_mean_loss_x1.index(min_x1)
            # new_network = list_network_x1[min_index_x1]
            # a = [1, -12, 35]
            # print(new_network.get_output(a))

            # for i in range(len(list_mean_loss_x1)):
            #     if list_mean_loss_x1[i] == min(list_mean_loss_x1)
                # if after_x2 < before_x2:
                #     for i in range(len(self.list_neural)):
                #         for j in range(len(self.list_neural[i])):
                #             min_neural[i][j].k[1] = self.list_neural[i][j].k[1]
                #     for i in range(len(self.list_edges)):
                #         for j in range(len(self.list_edges[i])):
                #             min_edge[i][j].weight[1] = self.list_edges[i][j].weight[1]

        # plt.subplot(121)
        # plt.title('График средней ошибок по x1')
        # x = np.arange(0, len(list_mean_loss_x1), 1)
        # plt.plot(x, list_mean_loss_x1)
        #
        # plt.subplot(122)
        # plt.title('График средней ошибок по x2')
        # x = np.arange(0, len(list_mean_loss_x2), 1)
        # plt.plot(x, list_mean_loss_x2)
        # plt.show()

        return Neural_network(min_neural, min_edge, self.learning_rate)


def progress_training(neural_net: Neural_network, condition=None, k=None):
    data_training = np.loadtxt('data.txt', dtype=int)
    list_mean_lost_x1 = []
    list_mean_lost_x2 = []
    min_neural = []
    min_neural_x1 = []
    min_edge_x1 = []
    min_edge = []
    list_network_x1 = []
    min_edge = []
    list_network_x2 = []

    list_network_x1.append(neural_net)

    neural_net.training_x1(data_training)
    list_network_x1.append(neural_net)

    print(list_network_x1[0].get_output([1, -3, 2]))
    print(list_network_x1[1].get_output([1, -3, 2]))

    # if k is None and condition is not None:
    #     while True:




# list_Neural = [
#             [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(1, 4)],
#             [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(4, 6)],
#             [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(6, 8)]]
# list_Edge = [[Edge(1, 4, np.array([random.random(), random.random()])),
#               Edge(1, 5, np.array([random.random(), random.random()])),
#               Edge(2, 5, np.array([random.random(), random.random()])),
#               Edge(3, 4, np.array([random.random(), random.random()])),
#               Edge(3, 5, np.array([random.random(), random.random()]))],
#              [Edge(4, 6, np.array([random.random(), random.random()])),
#               Edge(5, 6, np.array([random.random(), random.random()])),
#               Edge(5, 7, np.array([random.random(), random.random()]))]]

list_Neural = [[Neural(i, np.array([random.random(), random.random()])) for i in range(1, 4)],
               [Neural(i, np.array([random.random(), random.random()])) for i in range(4, 8)],
               [Neural(i, np.array([random.random(), random.random()])) for i in range(8, 11)],
               [Neural(i, np.array([random.random(), random.random()])) for i in range(11, 13)]]

list_Edge = [[Edge(1, 4, np.array([random.random() - 1, random.random() + 1])), Edge(1, 5, np.array([random.random() - 1, random.random() + 1])),
              Edge(1, 6, np.array([random.random() - 1, random.random() + 1])), Edge(2, 4, np.array([random.random(), random.random()])),
              Edge(2, 6, np.array([random.random(), random.random()])), Edge(2, 7, np.array([random.random(), random.random()])),
              Edge(3, 5, np.array([random.random(), random.random()])), Edge(3, 6, np.array([random.random(), random.random()])),
              Edge(3, 7, np.array([random.random(), random.random()]))],
             [Edge(4, 8, np.array([random.random(), random.random()])), Edge(4, 10, np.array([random.random(), random.random()])),
              Edge(5, 8, np.array([random.random(), random.random()])), Edge(5, 9, np.array([random.random(), random.random()])),
              Edge(6, 8, np.array([random.random(), random.random()])), Edge(6, 9, np.array([random.random(), random.random()])),
              Edge(6, 10, np.array([random.random(), random.random()])), Edge(7, 9, np.array([random.random(), random.random()])),
              Edge(7, 10, np.array([random.random(), random.random()]))],
             [Edge(8, 11, np.array([random.random(), random.random()])), Edge(8, 12, np.array([random.random(), random.random()])),
              Edge(9, 12, np.array([random.random(), random.random()])), Edge(10, 11, np.array([random.random(), random.random()])),
              Edge(10, 12, np.array([random.random(), random.random()]))]]
#
# list_Neural = [[Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(1, 4)],
#                [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(4, 7)],
#                [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(7, 11)],
#                [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(11, 14)],
#                [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(14, 16)]]
#
# list_Edge = [[Edge(1, 4, np.array([random.random(), random.random()])), Edge(1, 5, np.array([random.random(), random.random()])),
#               Edge(1, 6, np.array([random.random(), random.random()])), Edge(2, 4, np.array([random.random(), random.random()])),
#               Edge(2, 6, np.array([random.random(), random.random()])), Edge(3, 4, np.array([random.random(), random.random()])),
#               Edge(3, 5, np.array([random.random(), random.random()]))],
#              [Edge(4, 7, np.array([random.random(), random.random()])), Edge(4, 8, np.array([random.random(), random.random()])),
#               Edge(4, 9, np.array([random.random(), random.random()])), Edge(4, 10, np.array([random.random(), random.random()])),
#               Edge(5, 7, np.array([random.random(), random.random()])), Edge(5, 8, np.array([random.random(), random.random()])),
#               Edge(6, 7, np.array([random.random(), random.random()])), Edge(6, 8, np.array([random.random(), random.random()])),
#               Edge(6, 9, np.array([random.random(), random.random()])), Edge(6, 10, np.array([random.random(), random.random()]))],
#              [Edge(7, 11, np.array([random.random(), random.random()])), Edge(7, 12, np.array([random.random(), random.random()])),
#               Edge(7, 13, np.array([random.random(), random.random()])), Edge(8, 11, np.array([random.random(), random.random()])),
#               Edge(8, 12, np.array([random.random(), random.random()])), Edge(9, 11, np.array([random.random(), random.random()])),
#               Edge(9, 12, np.array([random.random(), random.random()])), Edge(9, 13, np.array([random.random(), random.random()])),
#               Edge(10, 12, np.array([random.random(), random.random()])), Edge(10, 13, np.array([random.random(), random.random()]))],
#              [Edge(11, 14, np.array([random.random(), random.random()])), Edge(11, 15, np.array([random.random(), random.random()])),
#               Edge(12, 14, np.array([random.random(), random.random()])), Edge(12, 15, np.array([random.random(), random.random()])),
#               Edge(13, 15, np.array([random.random(), random.random()]))]]

a = [1, -12, 35]
database_training = np.loadtxt('data.txt', dtype=int)

net = Neural_network(list_Neural, list_Edge, [0.05, 0.01])

progress_training(net)

# print(net.get_output(a))
# net.training_x1(database_training)
# print(net.get_output(a))
# net.training_x2(database_training)
# print(net.get_output(a))
#
# new_net = net.progress_training(k=100)
#
# print('actual x1 = ', net.get_output(a)[0], 'actual x2 = ', net.get_output(a)[1])
# # print('best x1 = ', new_net.get_output(a)[0], 'best x2 = ', new_net.get_output(a)[1])
# print('X1 = ', resolve_quadratic_equation(a)[0], 'X2 = ', resolve_quadratic_equation(a)[1])





