import sys

def getFileName():
    return sys.argv[1]

def removeEndLine(line):
    if(line.endswith("\n")):
        return line.replace("\n","")
    return line
 
def main():
    lines = open(getFileName(), 'r').readlines()
    lines = list(map(removeEndLine,lines))
    listOfLists = []
    for line in lines:
        splittedLine = line.split(",")
        listOfLists.append(splittedLine)
    print(listOfLists)

main()