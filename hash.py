class Record:
    def __init__(self, Key, Data):
        self.Key = Key
        self.Data = Data

HashTable = [[None for j in range(10)] for i in range(100)]

def InitialiseHashTable():
    global HashTable
    for i in range(100):
        for j in range(10):
            HashTable[i][j] = None

def Hash(Key):
    return Key % 100

def InsertData(Record):
    index = Hash(Record.Key)
    for j in range(10):
        if HashTable[index][j] is None:
            HashTable[index][j] = Record
            return
    
def ReadData():
    with open("HashTableData.txt", "r") as file:
        for line in file:
            parts = line.strip().split(",")
            key = int(parts[0])
            data = parts[1]
            newRecord = Record(key, data)
            InsertData(newRecord)
        file.close()

def GetRecord(Key):
    index = Hash(Key)
    for j in range(10):
        if HashTable[index][j] is not None:
            if HashTable[index][j].Key == Key:
                return HashTable[index][j].Data
    return "Not found"

InitialiseHashTable()
ReadData()

for i in range(5):
    key = int(input("Enter key : "))
    print(f"The value is{GetRecord(key)}" + "\n")