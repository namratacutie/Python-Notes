DataArray = []

with open("Data.txt", "r") as file:
    for lines in file:
        DataArray.append(int(lines.strip()))

def PrintArray(DataArray):
    output = ""

    for i in range(0, len(DataArray)):
        output = output + str(DataArray[i]) + " "

def LinearSearch(DataArray, target):

    count = 0

    for i in range(len(DataArray)):
        if target == DataArray[i]:
            count += 1
            
    print(f"\nThe number {target} is found {count} times.")


target = int(input("Enter the number between 0 - 100 : "))

PrintArray(DataArray)
LinearSearch(DataArray, target)

class Vehicle:
    def __init__(self, ID, MaxSpeed, CurrentSpeed, IncreaseAmount, HorizontalPosition):
        self.__ID = ID #Integer
        self.__MaxSpeed = MaxSpeed #Integer
        self.__CurrentSpeed = 0 #Integer
        self.__HorizontalPosition = 0 #Integer

    def GetCurrentSpeed(self):
        return self.__CurrentSpeed

    def GetIncreaseAmount(self):
        return self.__IncreaseAmount

    def GetMaxSpeed(self):
        return self.__MaxSpeed

    def GetHorizontalPosition(self):
        return self.__HorizontalPosition

    def SetCurrentSpeed(self, CurrentSpeed):
        self.__CurrentSpeed = CurrentSpeed

    def SetHorizontalPosition(self, HorizontalPosition):
        self.__HorizontalPosition = HorizontalPosition

    def IncreaseSpeed(self):
        CurrentSpeed += IncreaseAmount
        CurrentSpeed += HorizontalPosition