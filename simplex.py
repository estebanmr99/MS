# coding=utf-8

import sys
import numpy as np
from copy import deepcopy

M = 100
indexOfArtificialsBigM = []
outputFileName = ''

# Parameters: None
# Returns: None
# Description: The main structure for the program because begins and ends the different methods to solve problems of linear programming

def main():
    listOfLists = fileOperations()
    createOutputFile()
    loadValues(listOfLists)

# Parameters: None
# Returns: A matrix containing all the values read from the file sent as an argument
# Description: Splits the content of the file sent as an argument based on the commas

def fileOperations():
    lines = open(getFileName(), 'r').readlines()
    lines = list(map(removeEndLine, lines))
    listOfLists = []
    for line in lines:
        splittedLine = line.split(",")
        listOfLists.append(splittedLine)
    return listOfLists

# Parameters: None
# Returns: String - Name of the file sent in the arguments
# Description: Access to the stack to get the file name sent as an argument

def getFileName():
    return sys.argv[1]

# Parameters: String - Line from the file 
# Returns: Strnig - line without the newline char
# Description: Receives line by line the content from the file read and remove the newline char from each one

def removeEndLine(line):
    if(line.endswith("\n")):
        return line.replace("\n", "")
    return line

# Parameters: None
# Returns: None
# Description: Creates a new string from the name of the file adding the _solution.txt label and the file itself

def createOutputFile():
    fileName = getFileName()
    global outputFileName
    outputFileName = fileName.replace(".txt", "_solution.txt")
    open(outputFileName, 'w+')

# Parameters: Matrix, basicVariables - current basic variables in the iteration, pivotValues - contains the pivot number and pivot BV incoming and BV outgoing
#            , iteration - current iteration number
# Returns: None
# Description: Writes the current information of the iteration into the output file

def writeMatrixFile(matrix, basicVariables, pivotValues, iteration):
    testMatrix = np.matrix(matrix, dtype= np.str)

    aiv = []
    for i in range(len(matrix[0]) - 1):
        aiv.append(i + 1)
    aiv.append('RS')
    
    arrayInputValues = np.array(aiv,dtype=np.str)
    arrayInputValues = np.insert(arrayInputValues, 0, ['BV'], 0)

    arrayOutputValues = np.array(basicVariables, dtype=np.str)
    arrayOutputValues = np.insert(arrayOutputValues, 0, ['U'], 0)

    arrayIOP = np.array(pivotValues, dtype=np.str)

    a = np.insert(testMatrix, 0, arrayOutputValues, 1)

    b = np.insert(a, 0, arrayInputValues, 0)

    file = open(outputFileName, "a")

    file.write('Iteration: ' + str(iteration) + '\n')
    for row in b:
        for column in row:
            file.write(str(column) + ' ')
        file.write('\n')
    if(pivotValues):
        file.write('BV incoming: ' + str(arrayIOP[0]) + ', BV outgoing: ' + str(arrayIOP[1]) + ', Pivot number: ' + str(arrayIOP[2]) + '\n')
    file.write('\n')
    file.close()

# Parameters: text - it may be any string needed to be written in the output file and in the console
# Returns: None
# Description: Writes the string received as a parameter into the file

def writeInOuputFile(text):
    file =  open(outputFileName, "a")
    file.write(text + '\n')
    file.close
    print(text)

# Parameters: Matrix
# Returns: None
# Description: Handles the method to be used to solve the problem

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

# Parameters: optimization
# Returns: Matrix
# Description: if the optimization is 'max' multiply by -1 the first row of the matrix 

def mintoMax(optimization, matrix):
    if (optimization == 'max'):
        for variable in range(len(matrix[0])):
            matrix[0][variable] = num(matrix[0][variable]) * -1
    return matrix

# Parameters: matrix, optimizations, variables - int which represent the number of variables present in the problem
#           , restrictions - int which represent the number of restrictions present in the problem
# Returns: result - string containing the result of the problem
# Description: linear programing method using matrix operations

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

        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

        # divides the entire pivot row by the pivot number
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # does the operations between the rows and columns
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        iteration += 1
    roundMatrix(matrix)

    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # in case the optimization is 'min' it is multiplied by -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    isMultiplesolutions(matrix, basicVaribles)

    result = finalResult(matrix, matrixLen)
    
    return result

# Parameters: matrix, optimizations, variables - int which represent the number of variables present in the problem
#           , restrictions - int which represent the number of restrictions present in the problem
# Returns: result - string containing the result of the problem
# Description: linear programing method using matrix operations

def bigM(matrix, optimization, variables, restrictions):

    matrixLen = len(matrix)

    matrix = addNonBasicVariables(1, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    # handling of basic variables that are in the iteration
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

        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

        # divides the entire pivot row by the pivot number
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # does the operations between the rows and columns
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue

            idkHowtoCallIt = matrix[row][pivotColumn] * -1

            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        iteration += 1

    roundMatrix(matrix)

    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # in case the optimization is 'min' it is multiplied by -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    for i in range(len(basicVaribles)):
        if ((basicVaribles[i] - 1) in indexOfArtificialsBigM):
            writeInOuputFile('In the last iteration an artificial variable is still found so the solution is not feasible')

    isMultiplesolutions(matrix, basicVaribles)

    result = finalResult(matrix, matrixLen)

    text = 'With M = ' + str(M)
    writeInOuputFile(text)

    return result

# Parameters: matrix, optimizations, variables - int which represent the number of variables present in the problem
#           , restrictions - int which represent the number of restrictions present in the problem
# Returns: result - string containing the result of the problem
# Description: linear programing method using matrix operations

def twoPhases(matrix, optimization, variables, restrictions):
    matrixLen = len(matrix)

    copyMatrix = deepcopy(matrix)

    matrix = addNonBasicVariables(2, matrix, variables, restrictions)

    matrix = np.array([np.array(row) for row in matrix], dtype=object)

    # is an array that has the index where the artificial variables are
    artifialValuesIndex = (np.where(matrix[0] != 0))[0]

    # handling of basic variables that are in the iteration
    basicVaribles = []

    for i in range(len(artifialValuesIndex)):
        basicVaribles.append(artifialValuesIndex[i] + 1)

    count = 0

    for i in range((restrictions - len(basicVaribles))):
        basicVaribles.insert(count, variables + 1 + i)
        count += 1

    count = 0

    # prepares the objective function to be able to start with phase #1
    while(not isPrefase1Solution(matrix, artifialValuesIndex)):
        pivotRow = getPivotRowPrefase1(matrix, artifialValuesIndex)

        idkHowtoCallIt = matrix[0][artifialValuesIndex[count]] * -1

        test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

        matrix[0] = np.add(matrix[0], test)

        count += 1

    iteration = 0
    writeInOuputFile('Phase #1')

    # does phase #1 using the simplex algorithm
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

        # divides the entire pivot row by the pivot number
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # does the operations between the rows and columns
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        roundMatrix(matrix)

        iteration += 1

    
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    writeInOuputFile('Preparation phase #2')
    
    iteration = 0

    # step of eliminating artificial variables
    matrix = np.delete(matrix, artifialValuesIndex, 1)

    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # substitute the values in the objective function
    for objOrgValues in range(variables):
        matrix[0][objOrgValues] = num(copyMatrix[0][objOrgValues])

    if (optimization == 'max'):
        for variable in range(len(matrix[0])):
            matrix[0][variable] = int(matrix[0][variable]) * -1
    
    iteration += 1
    writeMatrixFile(matrix, basicVaribles, [], iteration)
    
    # zero basic variables
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

    # finishes phase 2 with the simplex algorithm
    while(isSolution(matrix)):
        pivotValues = getPivotValues(matrix)

        pivotNumber = pivotValues[0]
        pivotRow = pivotValues[1][0]
        pivotColumn = pivotValues[1][1]

        writeMatrixFile(matrix, basicVaribles, [(pivotColumn + 1), (basicVaribles[pivotRow - 1]), pivotNumber], iteration)

        basicVaribles[pivotRow - 1] = pivotColumn + 1

        # divides the entire pivot row by the pivot number
        matrix[pivotRow] = np.divide(matrix[pivotRow], pivotNumber)

        # does the operations between the rows and columns
        for row in range(matrixLen):
            roundMatrix(matrix)
            if (row == pivotRow or matrix[row][pivotColumn] == 0):
                continue
            
            idkHowtoCallIt = matrix[row][pivotColumn] * -1
            
            test = np.multiply(matrix[pivotRow], idkHowtoCallIt)

            matrix[row] = np.add(matrix[row], test)

        roundMatrix(matrix)
        iteration += 1

    
    writeMatrixFile(matrix, basicVaribles, [], iteration)

    # in case the optimization is 'min' it is multiplied by -1 U
    if(optimization == 'min'):
        matrix[0][len(matrix[0]) - 1] =  matrix[0][len(matrix[0]) - 1] * -1

    #obtiene el resultado EBNF de la ultima modificacion de la matriz 
    result =  finalResult(matrix,matrixLen)

    isMultiplesolutions(matrix, basicVaribles)

    return result

# Parameters: method -  represents the method selected by the user, matrix, 
#             variables - int which represent the number of variables present in the problem,
#             restrictions - int which represent the number of restrictions present in the problem
# Returns: matrix
# Description: prepares the matrix to be used on each method adding zeros and the ones to the artificials, excess, etc

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
        isArtificial = isPresentAnAritificial(matrix)

        if(isArtificial):
            count = 0
            equalRow = []
            excessColumn = []

            index = variables

            # matrix size adjustment and detection of special states for the Big M method

            for row in matrix:
                if(num(row[-1]) < 0):
                    row = row * -1
                    if('<=' in row):
                        row[-2] = '>='
                    elif('>=' in row):
                        row[-2] = '<='
                # if there is an '=' in the constraints, the row number is added to an array of rows with '='
                if ('=' in row):
                    equalRow.append(count)

                # if there is a '> =' in the constraints, the row number is added to an array of rows with '> ='
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

            matrix[0].append(0) # needed to add the right side value

            totalOfVaribles = len(matrix[0])

            # the array is traversed to add specific values
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

            # calculation that is made to row 0 due to variable M
            if(equalRow):
                for er in equalRow:
                    for v in range(len(matrix[0])):
                        matrix[0][v] = matrix[0][v] + (M * matrix[er][v] * -1)

            # modification made to the matrix, adding a new column
            if(excessColumn):
                for ex in excessColumn:
                    for e in range(len(matrix[0])):
                        matrix[0][e] = matrix[0][e] + (M * matrix[ex][e] * -1)
        else:
            writeInOuputFile('The problem doesn\'t contain any: \'>=\' or \'=\'')
            sys.exit()
                
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

            matrix[0].append(0) # Needed to add the right side value
            
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
        else:
            writeInOuputFile('The problem doesn\'t contain any: \'>=\' or \'=\'')
            sys.exit()
    return matrix

# Parameters: Matrix
# Returns: pivot values -  array containing the pivot row, pivot column and pivot number
# Description: determinates the pivot row, pivot column, and pivot number

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

        # checks for degenerate solutions at each iteration
        if(rigthSide == pivotValues[0]):
            writeInOuputFile("The solution is degenerate, the numbers in the pivot column that generate the case are: " + str(pivotNumber) + " and " + str(posiblePivotNumber))
        if(rigthSide < pivotValues[0]):
            pivotValues[0] = rigthSide
            pivotValues[1] = row
            pivotNumber = posiblePivotNumber
    
    # checks if the U is not bounded
    if(pivotNumber == 0):
        writeInOuputFile('In the current iteration all the values in the pivot column are negative or zero, therefore the U is not bounded')
        sys.exit()
    
    pivotValues[0] = pivotNumber
    pivotValues[1] = [pivotValues[1], indexOfLowestNumber]
    return pivotValues

# Parameters: Matrix
# Returns: Matrix
# Description: Removes the signs from the matrix and converts all the values into ints/floats

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

# Parameters: s - string that may be an float or int
# Returns: int or float
# Description: converts a string into float or int

def num(s):
    try:
        return float(s)
    except ValueError:
        return int(s)

# Parameters: matrix
# Returns: matrix
# Description: rounds off each value in the matrix to 5 digits if it is a float number

def roundMatrix(matrix):
    for row in range(len(matrix)):
        for column in range((len(matrix[row]))):
            matrix[row][column] = round(matrix[row][column],5)

    return matrix

# Parameters: Matrix
# Returns: Boleean
# Description: Determinates if in the current matrix exist a solution for the simplex algorithm 

def isSolution(matrix):
    lowestNumber = np.sort(matrix[0])
    indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[0]))[0][0]

    if (indexOfLowestNumber == (len(matrix[0]) - 1)):
        indexOfLowestNumber = (np.where(matrix[0] == lowestNumber[1]))[0][0]

    if(matrix[0][indexOfLowestNumber] >= 0 ):
        return False

    return True

# Parameters: matrix, matrixlen
# Returns: result - array containing the format for the final result
# Description: Based on the matrix returns an array with the values necessary for the BNF solution

def finalResult(matrix, matrixLen):
    result = []
    for i in range(matrixLen):
        if (i == 0):
            result.append(matrix[i][-1])
            result.append([])
        else:
            result[1].append(matrix[i][-1])

    return result

# Parameters: matrix, basicVariables - array containing the column index where the BV are placed
# Returns: matrix
# Description: Determinates if the problem has multiple solutions

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

# Parameters: Matrix, vararibles - index where the original values of the U function are placed
# Returns: Boolean
# Description: verifies that the basic variables have become zero in phase 2

def isfase2Solution(matrix, variables):
    isReady = False
    for i in range(variables):
        if (matrix[0][i] == 0):
            isReady = True
        else:
            isReady = False
    return isReady

# Parameters: Matrix, vararibles - index where the original values of the U function are placed
# Returns: Int - represent the row where the pivot number is placed
# Description: gets the row that has a 1 to be able to zero the basic variables of phase 2

def getPivotRowfase2(matrix, variables):
    matrixLen = len(matrix)

    for row in range(1, matrixLen):
        for i in range(variables):
            if(matrix[row][i] == 1 and matrix[0][i] != 0):
                return row

# Parameters: Matrix, artificialValesIndex - array that contains the column index where the artificial variables are placed
# Returns: Boleean
# Description: verifies that the artificial variables are set to zero, as long as they are not zero returns False

def isPrefase1Solution(matrix, artificialValuesIndex):
    isReady = False
    for i in artificialValuesIndex:
        if (matrix[0][i] != 0):
            isReady = False
        else:
            isReady = True

    return isReady

# Parameters: Matrix, artificialValesIndex - array that contains the column index where the artificial variables are placed
# Returns: Int - represent the row where the pivot number is placed
# Description: gets the row that has a 1 in the constraints to be able to zero the artificial variables in the objective function

def getPivotRowPrefase1(matrix, artifialValuesIndex):
    matrixLen = len(matrix)

    for row in range(1, matrixLen):
        for i in range(len(artifialValuesIndex)):
            if(matrix[row][artifialValuesIndex[i]] == 1 and matrix[0][artifialValuesIndex[i]] != 0):
                return row

# Parameters: Matrix
# Returns: Boleean
# Description: Determinates if an artificial variable is going to be present in the current matrix

def isPresentAnAritificial(matrix):
    for row in matrix:
        if ('>=' in row):
            return True
        elif ('=' in row):
            return True
    return False

main()