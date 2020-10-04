import sys
import numpy as np

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

    if (optimization == 'min'):
        matrix = mintoMax(matrix)

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

    print(result)


def mintoMax(matrix):
    for variable in range(len(matrix[0])):
        matrix[0][variable] = int(matrix[0][variable]) * -1
    return matrix


def simplex(matrix, optimization, variables, restrictions):

    matrix = addNonBasicVariables(0, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix])

    print(matrix)
    return 'nothig'


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