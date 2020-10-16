import sys
import numpy as np
from copy import deepcopy

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

    result = ''

    if (method == 0):
        matrix = mintoMax(optimization, matrix)
        result = simplex(matrix, optimization, variables, restrictions)

    elif (method == 1):
        matrix = mintoMax(optimization, matrix)
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

    #aqui se tiene que guardar la matriz

    # en caso de que la optimizacion sea 'min' se  multiplica por -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    print(matrix)

    result = finalResult(matrix, matrixLen)
    
    return result

def searchEqual(matrix):
    matrixLen = len(matrix)
    lineCount = 0
    for row in range(matrixLen):
        lineLen = len(row)
        if (row[lineLen - 1] == "="):
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
    lowestNumber = np.sort(matrix[0])
    indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[0]))[0][0]

    if (indexOfLowestNumber == (len(matrix[0]) - 1)):
        indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[1]))[0][0]

    if(matrix[0][indexOfLowestNumber] >= 0 ):
        return False

    return True


def getPivotValues(matrix):
    lowestNumber = np.sort(matrix[0])
    indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[0]))[0][0]

    if (indexOfLowestNumber == (len(matrix[0]) - 1)):
        indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[1]))[0][0]

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


def twoPhases(matrix, optimization, variables, restrictions):
    matrixLen = len(matrix)

    copyMatrix = deepcopy(matrix)

    matrix = addNonBasicVariables(2, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    if (optimization == 'max'):
        for variable in range(len(matrix[0])):
            matrix[0][variable] = int(matrix[0][variable]) * -1

    # aqui hay que guardar en el archivo

    #es un arreglo que posee el indice donde estan las variables artificiales
    artifialValuesIndex = (np.where(matrix[0] != 0))[0]

    count = 0

    #prepara la funcion objetivo para poder comenzar con la fase 1
    while(not isPrefase1Solution(matrix, artifialValuesIndex)):
        pivotRow = getPivotRowPrefase1(matrix, artifialValuesIndex)

        idkHowtoCallIt = matrix[0][artifialValuesIndex[count]] * -1

        test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

        matrix[0] = np.add(matrix[0], test)

        count += 1

    # aqui hay que guardar en el archivo

    # hace la fase 1 utilizando el algoritmo del simplex
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz

        # divide toda la fila pivot por el numero pivot
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # hace las operaciones entre las filas y columnas
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        roundMatrix(matrix)

    #paso de eliminar las variables artificiales
    matrix = np.delete(matrix, artifialValuesIndex, 1)

    # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz

    # sustituir los valores en la función objetivo
    for objOrgValues in range(variables):
        matrix[0][objOrgValues] = num(copyMatrix[0][objOrgValues])
    
    # hacer cero las variables básicas
    count = 0

    while(not isfase2Solution(matrix, variables)):
        roundMatrix(matrix)
        pivotRow = getPivotRowfase2(matrix, variables)

        idkHowtoCallIt = matrix[0][count] * -1

        test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

        matrix[0] = np.add(matrix[0], test)

        count += 1

    roundMatrix(matrix)

    # termina la fase 2 con el algoritmo de simplex
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz

        # divide toda la fila pivot por el numero pivot
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # hace las operaciones entre las filas y columnas
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        roundMatrix(matrix)

    #aqui se tiene que guardar la matriz

    # en caso de que la optimizacion sea 'min' se  multiplica por -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    #obtiene el resultado EBNF de la ultima modificacion de la matriz 
    result =  finalResult(matrix,matrixLen)

    print(matrix)

    return result

# verifica que las variables basicas se hayan vuelto cero en la fase 2
def isfase2Solution(matrix, variables):
    isReady = False
    for i in range(variables):
        if (matrix[0][i] == 0):
            isReady = True
        else:
            isReady = False
    return isReady

# obtiene la fila que posee un 1 para poder hacer cero las variables basicas de la fase 2
def getPivotRowfase2(matrix, variables):
    matrixLen = len(matrix)

    for row in range(1, matrixLen):
        for i in range(variables):
            if(matrix[row][i] == 1 and matrix[0][i] != 0):
                return row


#verifica que se hagan cero las variables atificiales, mientras no sea sean cero devuelve False
def isPrefase1Solution(matrix, artificialValuesIndex):
    isReady = False
    for i in artificialValuesIndex:
        if (matrix[0][i] != 0):
            isReady = False
        else:
            isReady = True

    return isReady

#cada valor en la matrix es necesario redondearlo para que la precision no afecte, se redondea con 5 decimales 
def roundMatrix(matrix):
    for row in range(len(matrix)):
        for column in range((len(matrix[row]))):
            matrix[row][column] = round(matrix[row][column],5)

    return matrix

# obtiene la fila que posee un 1 en las restricciones para poder hacer cero las variables artificiales en la funcion objetivo
def getPivotRowPrefase1(matrix, artifialValuesIndex):
    matrixLen = len(matrix)

    for row in range(1, matrixLen):
        for i in range(len(artifialValuesIndex)):
            if(matrix[row][artifialValuesIndex[i]] == 1 and matrix[0][artifialValuesIndex[i]] != 0):
                return row
    

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
        count = 0

        isArtificial = isPresentAnAritificial(matrix)

        if(isArtificial):
            orgSizeObj =  len(matrix[0])

            for row in range(1, len(matrix)):
                if ('=' in matrix[row]):
                    matrix[0].append(1)
                elif('>=' in matrix[row]):
                    matrix[0].append(0)
                    matrix[0].append(1)
                else:
                    matrix[0].append(0)

            matrix[0].append(0) # Necesario para agregar el valor de lado derecho
            
            for i in range(orgSizeObj):
                matrix[0][i] = 0
        
            count = 1

            totalOfVaribles = len(matrix[0])

            for row in range(len(matrix)):
                for column in range(totalOfVaribles):
                    if (row >= 1 and column > variables):                        
                        matrix[row].insert((len(matrix[row]) - 1), 0)
                if (row >= 1):
                    if(matrix[row][variables] == '>='):
                        matrix[row][variables + count] = -1
                        count +=1
                        matrix[row][variables + count] = 1
                    else:
                        matrix[row][variables + count] = 1
                    count += 1

            matrix = formatMatrix(matrix)
    return matrix

def isPresentAnAritificial(matrix):
    for row in matrix:
        if ('>=' in row):
            return True
        elif ('=' in row):
            return True
    return False

def formatMatrix(matrix):
    for row in range(len(matrix)):
        if ('<=' in matrix[row]):
            matrix[row].remove('<=')
        elif ('>=' in matrix[row]):
            matrix[row].remove('>=')
        elif ('=' in matrix[row]):
            matrix[row].remove('=')
        for column in range(len(matrix[row])):
            matrix[row][column] = num(matrix[row][column])
            
    return matrix

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)
def main():
    listOfLists = fileOperations()
    loadValues(listOfLists)


main()