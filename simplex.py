import numpy as np

def simplex(tableu: np.ndarray, zeroIndexed=True, printIntermediateTableu=False) -> np.ndarray:

    z = tableu[0, 1:-1]
    basic = tableu[1:, 0]

    cost = tableu[:, -1]

    matrix = tableu[1:, 1:-1]
    
    if printIntermediateTableu: print(np.c_[tableu[:,0] + int(zeroIndexed),tableu[:,1:]])

    if np.all(z <= 0):
        # overallCost = tableu[0,-1]
        return basic + int(zeroIndexed), cost[1:] , cost[0]
    else:
        
        indIn = np.argmax(z[:-1])
        indOut = np.argmin(cost[1:] / matrix[:, indIn])
        basic[indOut] = indIn
        
        extMatrix = tableu[:,1:]
        extMatrix[indOut + 1] /= extMatrix[indOut + 1, indIn]

        for i in range(len(tableu)):
            if i == indOut + 1:
                continue
            else:
                extMatrix[i] -= extMatrix[indOut + 1] * extMatrix[i, indIn]
        
        # cost[1:] -= cost[indOut] * matrix[indOut]
        return simplex(tableu, zeroIndexed, printIntermediateTableu)
        # tableu
    # pass


np.set_printoptions(precision=4, suppress=True, linewidth=10000)

if __name__ == "__main__":
    t = np.array([
        [0,6,4,3,0,0,0,0],
        [3,4,5,3,1,0,0,12],
        [4,3,4,2,0,1,0,10],
        [5,4,2,1,0,0,1,8]
    ], dtype=np.float128)
    basic, cost, c = simplex(t, printIntermediateTableu=True)

    print(basic, cost, c)
