import sys
import numpy as np

M = 100

def getFileName():
    return sys.argv[1]

def removeEndLine(line):
    if(line.endswith("\n")):
        return line.replace("\n", "")
    return line

def fileOperations():
    lines = open(getFileName(), 'r').readlines()
    lines = list(map(removeEndLine, lines))
    listOfLists = []
    for line in lines:
        splittedLine = line.split(",")
        listOfLists.append(splittedLine)
    return listOfLists

def loadValues(matrix):
    fistLine = matrix.pop(0)
    method = int(fistLine[0])
    optimization = fistLine[1]
    variables = int(fistLine[2])
    restrictions = int(fistLine[3])

    
    matrix = mintoMax(optimization, matrix)

    result = ''

    if (method == 0):
        result = simplex(matrix, optimization, variables, restrictions)

    elif (method == 1):
        result = bigM(matrix, optimization, variables, restrictions)

    elif (method == 2):
        result = twoPhases(matrix, optimization, variables, restrictions)

    else:
        print('A non a valid method was sent as parameter')
        return None

    print("The final result is: U = " + str(result[0]) + ", " + str(result[1]))


def mintoMax(optimization, matrix):
    if (optimization == 'max'):
        for variable in range(len(matrix[0])):
            matrix[0][variable] = int(matrix[0][variable]) * -1
    return matrix


def simplex(matrix, optimization, variables, restrictions):

    matrixLen = len(matrix)

    matrix = addNonBasicVariables(0, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz

        #divide toda la fila pivot por el numero pivot
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # hace las operaciones entre las filas y columnas
        for row in range(matrixLen):
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

    result = finalResult(matrix, matrixLen)
    
    return result

def searchEqual(matrix):
    matrixLen = len(matrix)
    lineCount = 0
    for row in range(matrixLen):
        lineLen = len(row)
        if (row[lineLen - 1] == "="):
            return lineCount
            return lineCount
        lineCount += 1

def bigM(matrix, optimization, variables, restrictions):

    restrictLine = searchEqual(matrix)
    matrixLen = len(matrix)

    matrix = addNonBasicVariables(0, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    for v in range(variables):
        matrix[0][v] = matrix[0][v] + (M * matrix[restrictLine][v])
    matrix[0][matrixLen-1] = matrix[0][matrixLen-1] + (M * matrix[restrictLine][matrixLen-1])

    while (isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz

        # divide toda la fila pivot por el numero pivot
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # hace las operaciones entre las filas y columnas
        for row in range(matrixLen):
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue

            idkHowtoCallIt = matrix[row][pivotColumn] * -1

            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

    result = finalResult(matrix, matrixLen)

    return result


def finalResult(matrix, matrixLen):
    result = []
    for i in range(matrixLen):
        if (i == 0):
            result.append(matrix[i][-1])
            result.append([])
        else:
            result[1].append(matrix[i][-1])

    return result


def isSolution(matrix):
    lowestNumber = np.amin(matrix[0])

    if(lowestNumber >= 0 ):
        return False

    return True


def getPivotValues(matrix):
    lowestNumber = np.amin(matrix[0])
    indexOfLowestNumber = (np.where(matrix[0] == lowestNumber))[0][0]

    pivotValues = []
    matrixLen = len(matrix)
    rowLen = len(matrix[0]) - 1
    pivotNumber = 0

    for row in range(1, matrixLen):
        posiblePivotNumber = matrix[row][indexOfLowestNumber]
        if (posiblePivotNumber <= 0):
            continue
        rigthSide =  matrix[row][rowLen] / posiblePivotNumber
        if (len(pivotValues) == 0):
            pivotValues.insert(0, rigthSide)
            pivotValues.insert(1, row)
            pivotNumber = posiblePivotNumber
        if(rigthSide < pivotValues[0]):
            pivotValues[0] = rigthSide
            pivotValues[1] = row
            pivotNumber = posiblePivotNumber
    
    pivotValues[0] = pivotNumber
    pivotValues[1] = [pivotValues[1], indexOfLowestNumber]
    return pivotValues


def bigM(matrix, optimization, variables, restrictions):
    return 'nothing'


def twoPhases(matrix, optimization, variables, restrictions):
    return 'nothing'


def addNonBasicVariables(method, matrix, variables, restrictions):
    if (method == 0):
        count = 0

        for row in matrix:
            if ('>=' in row or '=' in row):
                print('A non valid operator was sent in the values')
                sys.exit()

            if (count <= restrictions):
                matrix[0].append(0)
                count += 1

        matrix = formatMatrix(matrix)
        count = 0

        totalOfVaribles = len(matrix[0])

        for row in range(len(matrix)):
            for column in range(totalOfVaribles):
                if (row >= 1 and column > variables):
                    matrix[row].insert((len(matrix[row]) - 1), 0)
            if (row >= 1):
                matrix[row][variables + count] = 1
                count += 1


    elif (method == 1):
        test = 1

    elif (method == 2):
        test = 2

    return matrix

def formatMatrix(matrix):
    for row in range(len(matrix)):
        if ('<=' in matrix[row]):
            matrix[row].remove('<=')
        elif ('>=' in matrix[row]):
            matrix[row].remove('>=')
        elif ('=' in matrix[row]):
            matrix[row].remove('=')
        for column in range(len(matrix[row])):
            matrix[row][column] = int(matrix[row][column])
            
    return matrix

def main():
    listOfLists = fileOperations()
    loadValues(listOfLists)


main()