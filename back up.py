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
    def __init__(self, id: int, input_data: [], k: []):
        self.id = id
        self.input_data = input_data.copy()
        self.k = k.copy()

    def get_output(self):
        return [self.input_data[0] * self.k[0], self.input_data[1] * self.k[1]]


class Edge:
    def __init__(self, left: int, right: int, weight: []):
        self.left = left
        self.right = right
        self.weight = weight.copy()


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
                    j.input_data[1] += list_neural_to_this_neural[k].get_output()[1] * \
                                       list_edges_to_this_neural[k].weight[1]

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
        for i in range(start_layer, 2):
            for j in range(len(road)):
                k = len(road[0])
                next_neural = self.find_right_neighbour_neural(road[0][k - 1])
                for x in next_neural:
                    copy = road[0].copy()
                    copy.append(x)
                    road.append(copy)
                road.remove(road[0])
        result = []
        for i in road:
            if i[-1].id == last_neural.id:
                result.append(i)
        return result

    def get_loss_x1(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        return (actual_resolve[0] - expected_resolve[0])**2

    def get_loss_x2(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        return (actual_resolve[1] - expected_resolve[1]) ** 2

    def dl1_by_dx1(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        return 2 * (actual_resolve[0] - expected_resolve[0])

    def dl2_by_dx2(self, input_para: []):
        expected_resolve = resolve_quadratic_equation(input_para)
        actual_resolve = self.get_output(input_para)
        return 2 * (actual_resolve[1] - expected_resolve[1])

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
        print(next_neural.id)
        road = self.find_road(next_neural, self.find_neural(7))
        print(len(road))
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

            # if i == 1:
            #     print('actual x1 = ', self.get_output(data_training[i])[0], 'actual x2 = ',
            #           self.get_output(data_training[i])[1])
            #     print('expected x1 = ', resolve_quadratic_equation(data_training[i])[0], 'expected x2 = ',
            #           resolve_quadratic_equation(data_training[i])[1])
            #     print(dl1_by_dw1)
            #     print(dl1_by_dk1)

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
            # print('actual x1 = ', self.get_output(data_training[i])[0], 'X1 = ', resolve_quadratic_equation(data_training[i])[0], 'loss = ', self.get_loss_x1(data_training[i]))
        mean_loss_after = sum(losses_after)/len(losses_after)
        return mean_loss_after

    def training_x2(self, data_training: []):
        expected_x2 = []
        for i in data_training:
            expected_x2.append(resolve_quadratic_equation(data_training[i]))
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
        return mean_loss_after


list_Neural = [
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(1, 4)],
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(4, 6)],
            [Neural(i, np.array([0.0, 0.0]), np.array([random.random(), random.random()])) for i in range(6, 8)]]
list_Edge = [[Edge(1, 4, np.array([random.random(), random.random()])),
              Edge(1, 5, np.array([random.random(), random.random()])),
              Edge(2, 5, np.array([random.random(), random.random()])),
              Edge(3, 4, np.array([random.random(), random.random()])),
              Edge(3, 5, np.array([random.random(), random.random()]))],
             [Edge(4, 6, np.array([random.random(), random.random()])),
              Edge(5, 6, np.array([random.random(), random.random()])),
              Edge(5, 7, np.array([random.random(), random.random()]))]]

a = [1, -3, 2]
database_training = np.loadtxt('data.txt', dtype=int)
net = Neural_network(list_Neural, list_Edge, [0.05, 0.05])

list_mean_loss_x1 = []
for i in range(1000):
    after = net.training_x1(database_training)
    list_mean_loss_x1.append(after)

x = np.arange(0, 1000, 1)
plt.plot(x, list_mean_loss_x1)
plt.show()






