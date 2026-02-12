arrayData = [10, 5, 6, 7, 1, 12, 13, 15, 21, 8]
target = int(input("Enter the input integer : "))
found = False

def linearSearch(target):
    global arrayData
    global found

    for i in range(len(arrayData)):
            if target == arrayData[i]:
                print(f"The targeted value {target} found in the index {i}")
                found = True
                break

    if not found:
        print("The targeted integer not found in the array.")


def bubbleSort():
    global arrayData
    temp = 0

    for x in range(len(arrayData)):
        for y in range(len(arrayData) - x - 1):
            if arrayData[y] > arrayData[y + 1]:
                temp = arrayData[y]
                arrayData[y] = arrayData[y + 1]
                arrayData[y + 1] = temp

linearSearch(target)
bubbleSort()
print(arrayData)

