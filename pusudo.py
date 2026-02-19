AnimalTopPointer = 0
ColorTopPointer = 0
Animal = []
Color = []

def PushAnimal(DataToPush):
    global AnimalTopPointer

    if AnimalTopPointer == 20:
        return False

    else:
        AnimalTopPointer += 1
        Animal.append(DataToPush)
        return True

def PushColor(DataToPush):
    global ColorTopPointer

    if ColorTopPointer == 20:
        return False

    else:
        ColorTopPointer += 1
        Color.append(DataToPush)
        return True

def PopAnimal(DataToPop):
    global AnimalTopPointer
    ReturnData = ""

    if AnimalTopPointer == 0:
        return ""

    else:
        ReturnData = Animal[AnimalTopPointer - 1]
        AnimalTopPointer -= 1
        return ReturnData

def PopColor(DataToPop):
    global ColorTopPointer
    ReturnData = ""

    if ColorTopPointer == 0:
        return ""

    else:
        ReturnData = Color[ColorTopPointer - 1]
        ColorTopPointer += 1
        return ReturnData


def ReadData():
    try:
        with open("AnimalData.txt", "r") as AnimalFile:
            for AnimalLines in AnimalFile:
                PushAnimal(AnimalLines.strip())

        with open("ColorData.txt", "r") as ColorFile:
            for ColorLines in ColorFile:
                PushColor(ColorLines.strip())

    except FileNotFoundError:
        print("File does not exist")

def OutputItem():
    global AnimalTopPointer
    global ColorTopPointer

    ReturnedAnimal = PopAnimal
    ReturnedColor = PopColor

    if ReturnedColor == "":
        print("No color")
        PushAnimal(ReturnedAnimal)

    else:
        if ReturnedAnimal == "":
            print("No animal")
            PushColor(ReturnedColor)

        else:
            print(ReturnedColor, ReturnedAnimal)

ReadData()
OutputItem()
