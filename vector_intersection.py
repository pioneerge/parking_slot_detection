import numpy as np
import descartes
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString


def point_in_rectangle(point, rectangle):
    AB = (rectangle[0], rectangle[1])
    BC = (rectangle[1], rectangle[2])
    AM = (rectangle[0], point)
    BM = (rectangle[1], point)
    dotABAM = np.dot(AB, AM)
    dotABAB = np.dot(AB, AB)
    dotBCBM = np.dot(BC, BM)
    dotBCBC = np.dot(BC, BC)
    print(dotABAM, dotABAB, " -- ", dotBCBM, dotBCBC)
    return 0 <= dotABAM <= dotABAB and 0 <= dotBCBM <= dotBCBC


def vector_in_polygon(vector, polygon, draw=False):
    vector = LineString(vector)
    polygon = Polygon(polygon)

    # Drawing function
    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(*np.array(vector).T, color='blue', linewidth=3, solid_capstyle='round')
        ax.add_patch(descartes.PolygonPatch(polygon, fc='blue', alpha=0.5))
        ax.axis('equal')
        plt.show()

    return vector.intersects(polygon)


def vector_intersection(vector, cars, height, width):
    """
    :param vector: [point_begin, point_end]
    :param cars: [[[x0, y0], .., [x4, y4]], ...]
    :return:  True/Flase, Car coords
        - If True: coords of parking space
        - If False: coords of car that is in the way
    """
    for car in cars:
        left_vector = vector  # left right points - for the corners of the car
        right_vector = vector
        left_vector[1][0] -= width / 2
        right_vector[1][0] += width / 2
        if vector_in_polygon(vector, car.copy()):
            return False, car

    # Calculating place for parking
    y_middle = (vector[1][1] - vector[0][1]) / 2
    x_middle = (vector[1][0] - vector[0][0]) / 2
    tl = [x_middle - width / 2, y_middle + height / 2]
    tr = [x_middle + width / 2, y_middle + height / 2]
    bl = [x_middle - width / 2, y_middle - height / 2]
    br = [x_middle + width / 2, y_middle - height / 2]

    return True, [tl, tr, br, bl]


if __name__ == '__main__':
    rec = [[[0, 0],
           [0, 2],
           [1, 2],
           [1, 0],
           [0, 0]],

           [[1, 5],
            [0, 5],
            [0, 7],
            [1, 7],
            [1, 5]]]

    vector = [[0.5, 2.1], [0.5, 4]]
    print(rec[0])
    # print(vector_in_polygon(vector, rec[0]))
    print(vector_intersection(vector, rec, width=1, height=2))
