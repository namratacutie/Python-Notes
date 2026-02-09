def ReadData():
    fileName = input("Enter the file name : ")
    dataList = []

    try:
        with open(fileName, "r") as file:
            for line in file:
                dataList.append(line.strip())
    except:
        print("Error while reading the file")

    return dataList

def SplitData():
    Red = []
    Blue = []
    Green = []
    Orange = []
    Yellow = []
    Pink = []

    for items in dataList:
        parts = items.split(",")
        number = int(parts[0])
        color = parts[1].lower()

        if colour == "red":
            Red.append(number)

        elif colour == "blue":
            Blue.append(number)

        elif colour == "green":
            Green.append(number)

        elif colour == "orange":
            Orange.append(number)

        elif colour == "yellow":
            Yellow.append(number)

        elif colour == "pink":
            Pink.append(number)

def StoreData(DataToStore, fileName):
    try:
        with open(fileName, "a") as file:
            for items in DataToStore:
                file.write(str(items) + "\n")

    except:
        print(f"Error while writing the data")

StoreData(Red, "Red.txt")
StoreData(Blue, "Blue.txt")
StoreData(Green, "Green.txt")
StoreData(Orange, "Orange.txt")
StoreData(Yellow, "Yellow.txt")
StoreData(Pink, "Pink.txt")

Data = ReadData()
SplitData(Data)
print("Data processing complete.")
