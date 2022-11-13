import numpy as np


def swapIn(tableu: np.ndarray, indIn: int ,indOut: int):

    basic = tableu[1:,0]        
    basic[indOut] = indIn

    extMatrix = tableu[:,1:]
    extMatrix[indOut + 1] /= extMatrix[indOut + 1, indIn]

    for i in range(len(tableu)):
        if i == indOut + 1:
            continue
        else:
            extMatrix[i] -= extMatrix[indOut + 1] * extMatrix[i, indIn]
                

def simplex(tableu: np.ndarray, zeroIndexed=True, printIntermediateTableu=False, blandsRule=False) -> np.ndarray:

    z = tableu[0, 1:-1]
    basic = tableu[1:, 0]

    cost = tableu[:, -1]

    matrix = tableu[1:, 1:-1]
    
    if printIntermediateTableu: print(np.c_[tableu[:,0] + int(zeroIndexed),tableu[:,1:]])

    if np.all(z <= 0):
        # overallCost = tableu[0,-1]
        return basic + int(zeroIndexed), cost[1:] , cost[0]
    else:
        if not blandsRule:
            indIn = np.argsort(z[:-1])[::-1]
            for ind in indIn: 
                arr = [(cost[k+1] / i if i > 0 else np.inf) for k, i in enumerate(matrix[:, ind])]
                if z[:-1][ind] <= 0:
                    return basic + int(zeroIndexed), cost[1:] , cost[0]
                if np.all(np.array(arr) == np.inf):
                    continue
                    # return basic + int(zeroIndexed), cost[1:] , cost[0]
                else:
                    break
            indOut = np.argmin(arr)
            indIn = ind
            
            swapIn(tableu, indIn, indOut)
            
            return simplex(tableu, zeroIndexed, printIntermediateTableu)
        
        else:
            indIn = range(len(z[:-1]))
            for ind in indIn: 
                arr = [(cost[k+1] / i if i > 0 else np.inf) for k, i in enumerate(matrix[:, ind])]
                if z[:-1][ind] <= 0:
                    continue
                if np.all(np.array(arr) == np.inf):
                    continue
                    # return basic + int(zeroIndexed), cost[1:] , cost[0]
                else:
                    break

            if z[:-1][ind] <= 0:
                return basic + int(zeroIndexed), cost[1:] , cost[0]
                
            indOut = np.argmin(arr)
            indIn = ind
            swapIn(tableu, indIn, indOut)

            return simplex(tableu, zeroIndexed, printIntermediateTableu, blandsRule)
        

# np.set_printoptions(precision=4, suppress=True, linewidth=10000)

if __name__ == "__main__":
    pass


    