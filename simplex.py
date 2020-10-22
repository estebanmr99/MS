# coding=utf-8

import sys
import numpy as np
from copy import deepcopy

M = 100
indexOfArtificialsBigM = []
outputFileName = ''

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

    writeInOuputFile("The final result is: U = " + str(result[0]) + ", " + str(result[1]))


def mintoMax(optimization, matrix):
    if (optimization == 'max'):
        for variable in range(len(matrix[0])):
            matrix[0][variable] = num(matrix[0][variable]) * -1
    return matrix

def isMultiplesolutions(matrix, basicVariables):
    nonBasicVariables = []
    lenURow = len(matrix[0]) - 1

    for i in range(lenURow):
        if (not (i + 1) in basicVariables):
            nonBasicVariables.append(i)

    for i in range(len(nonBasicVariables)):
        if  (-0.0001 <= matrix[0][nonBasicVariables[i]] <= 0.0001):
            writeInOuputFile("This problem has multiple solutions")

    return matrix


def simplex(matrix, optimization, variables, restrictions):

    matrixLen = len(matrix)

    matrix = addNonBasicVariables(0, matrix, variables, restrictions)

    basicVaribles = []

    for i in range(variables + 1, (len(matrix[0]))):
        basicVaribles.append(i)

    iteration = 0

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt
        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

        #divide toda la fila pivot por el numero pivot
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # hace las operaciones entre las filas y columnas
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        iteration += 1
    roundMatrix(matrix)
    #aqui se tiene que guardar la matriz
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # en caso de que la optimizacion sea 'min' se  multiplica por -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    isMultiplesolutions(matrix, basicVaribles)

    result = finalResult(matrix, matrixLen)
    
    return result

def bigM(matrix, optimization, variables, restrictions):

    matrixLen = len(matrix)

    matrix = addNonBasicVariables(1, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    # manejo de varibles basicas que hay en la iteracion
    basicVaribles = []

    for i in range(len(indexOfArtificialsBigM)):
        basicVaribles.append(indexOfArtificialsBigM[i] + 1)

    count = 0

    for i in range((restrictions - len(basicVaribles))):
        basicVaribles.insert(count, variables + 1 + i)
        count += 1

    count = 0
    
    iteration = 0
    
    while (isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt
        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

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

        iteration += 1

    roundMatrix(matrix)

    #aqui se tiene que guardar la matriz
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # en caso de que la optimizacion sea 'min' se  multiplica por -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    for i in range(len(basicVaribles)):
        if ((basicVaribles[i] - 1) in indexOfArtificialsBigM):
            writeInOuputFile('In the last iteration an artificial variable is still found so the solution is not feasible')

    isMultiplesolutions(matrix, basicVaribles)

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
            continue

        #verifica si hay soluciones de generadas en cada iteración -- si funciona para simplex
        if(rigthSide == pivotValues[0]):
            writeInOuputFile("The solution is degenerate, the numbers in the pivot column that generate the case are:" + str(pivotNumber) + " and " + str(posiblePivotNumber))
        if(rigthSide < pivotValues[0]):
            pivotValues[0] = rigthSide
            pivotValues[1] = row
            pivotNumber = posiblePivotNumber
    
    #verifica la U no esta acotada -- si funciona para simplex
    if(pivotNumber == 0):
        writeInOuputFile('In the current iteration all the values in the pivot column are negative or zero, therefore the U is not bounded')
        sys.exit()
    
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

    # manejo de varibles basicas que hay en la iteracion
    basicVaribles = []

    for i in range(len(artifialValuesIndex)):
        basicVaribles.append(artifialValuesIndex[i] + 1)

    count = 0

    for i in range((restrictions - len(basicVaribles))):
        basicVaribles.insert(count, variables + 1 + i)
        count += 1

    count = 0

    #prepara la funcion objetivo para poder comenzar con la fase 1
    while(not isPrefase1Solution(matrix, artifialValuesIndex)):
        pivotRow = getPivotRowPrefase1(matrix, artifialValuesIndex)

        idkHowtoCallIt = matrix[0][artifialValuesIndex[count]] * -1

        test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

        matrix[0] = np.add(matrix[0], test)

        count += 1

    iteration = 0
    writeInOuputFile('Phase #1')

    # hace la fase 1 utilizando el algoritmo del simplex
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt
        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

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

        iteration += 1

    #aqui se tiene que guardar la matriz
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    writeInOuputFile('Preparation phase #2')
    
    iteration = 0

    #paso de eliminar las variables artificiales
    matrix = np.delete(matrix, artifialValuesIndex, 1)

    # Aqui deberia llamar a la funcion que guarda en el txt y la que verifica la matriz
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # sustituir los valores en la función objetivo
    for objOrgValues in range(variables):
        matrix[0][objOrgValues] = num(copyMatrix[0][objOrgValues])
    
    iteration += 1
    writeMatrixFile(matrix, basicVaribles, [], iteration)
    
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

    iteration += 1
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    iteration = 0
    writeInOuputFile('Phase #2')

    # termina la fase 2 con el algoritmo de simplex
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        # Aqui deberia llamar a la funcion que guarda en el txt
        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

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
        iteration += 1

    #aqui se tiene que guardar la matriz
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # en caso de que la optimizacion sea 'min' se  multiplica por -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    #obtiene el resultado EBNF de la ultima modificacion de la matriz 
    result =  finalResult(matrix,matrixLen)

    isMultiplesolutions(matrix, basicVaribles)

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

    #Ajuste de la matriz cuando se tiene que hacer el metodo simplex
    elif (method == 1):
        #definicion de variables a usar
        count = 0
        equalRow = []
        excessColumn = []

        # ------
        index = variables
        # ------

        #ajuste del tamaño de la matriz y deteccion de estados especiales para el metodo GranM
        for row in matrix:
            if(num(row[-1]) < 0):
                row = row * -1
                if('<=' in row):
                    row[-2] = '>='
                elif('>=' in row):
                    row[-2] = '<='
            #si en las restricciones hay un '=', se agrega el numero de fila a un arreglo de filas con '='
            if ('=' in row):
                equalRow.append(count)
            #si en las restricciones hay un '>=', se agrega el numero de fila a un arreglo de filas con '>='
            elif ('>=' in row):
                excessColumn.append(count)
            count+=1

        for row in range(1, len(matrix)):
            if ('=' in matrix[row]):
                matrix[0].append(M)
                indexOfArtificialsBigM.append(index)
                index += 1
            elif ('>=' in matrix[row]):
                matrix[0].append(0)
                matrix[0].append(M)
                index += 1
                indexOfArtificialsBigM.append(index)
                index += 1
            else:
                matrix[0].append(0)
                index += 1

        matrix[0].append(0)  # Necesario para agregar el valor de lado derecho

        totalOfVaribles = len(matrix[0])

        #se recorre la matriz para agregarle valores especificos
        count = 1

        totalOfVaribles = len(matrix[0])

        for row in range(len(matrix)):
            for column in range(totalOfVaribles):
                if (row >= 1 and column > variables):
                    matrix[row].insert((len(matrix[row]) - 1), 0)
            if (row >= 1):
                if (matrix[row][variables] == '>='):
                    matrix[row][variables + count] = -1
                    count += 1
                    matrix[row][variables + count] = 1
                else:
                    matrix[row][variables + count] = 1
                count += 1

        matrix = formatMatrix(matrix)

        #calculo que se realiza a la fila 0 debido a la variable M
        if(equalRow):
            for er in equalRow:
                for v in range(len(matrix[0])):
                    matrix[0][v] = matrix[0][v] + (M * matrix[er][v] * -1)

        #modificacion que se hace a la matriz, agregando una columna nueva
        if(excessColumn):
            for ex in excessColumn:
                for e in range(len(matrix[0])):
                    matrix[0][e] = matrix[0][e] + (M * matrix[ex][e] * -1)
        
                
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

def createOutputFile():
    fileName = getFileName()
    global outputFileName
    outputFileName = fileName.replace(".txt", "_solution.txt")
    open(outputFileName, 'w+')

def writeInOuputFile(text):
    file =  open(outputFileName, "a")
    file.write(text + '\n')
    file.close
    print(text)

def writeMatrixFile(s, aov, aiop, iteration):
    testMatrix = np.matrix(s, dtype= np.str)

    aiv = []
    for i in range(len(s[0]) - 1):
        aiv.append(i + 1)
    aiv.append('RS')
    
    arrayInputValues = np.array(aiv,dtype=np.str)
    arrayInputValues = np.insert(arrayInputValues, 0, ['BV'], 0)

    arrayOutputValues = np.array(aov, dtype=np.str)
    arrayOutputValues = np.insert(arrayOutputValues, 0, ['U'], 0)

    arrayIOP = np.array(aiop, dtype=np.str)


    a = np.insert(testMatrix, 0, arrayOutputValues, 1)

    b = np.insert(a, 0, arrayInputValues, 0)

    file = open(outputFileName, "a")

    file.write('Iteration: ' + str(iteration) + '\n')
    for row in b:
        for column in row:
            file.write(str(column) + ' ')
        file.write('\n')
    if(aiop):
        file.write('BV incoming: ' + str(arrayIOP[0]) + ', BV outgoing: ' + str(arrayIOP[1]) + ', Pivot number: ' + str(arrayIOP[2]) + '\n')
    file.write('\n')
    file.close()


def num(s):
    try:
        return float(s)
    except ValueError:
        return int(s)

def main():
    listOfLists = fileOperations()
    createOutputFile()
    loadValues(listOfLists)

main()