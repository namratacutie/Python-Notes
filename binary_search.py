arr = [3,23,3,234,2,34,32423,23,213,4,345,5,8,876,745,234,123,234,456,58,5,44423,4234,423,234,23,432,54,57]
found = False
target = 745
temp = 0

def bubble_sort():
    for i in range(len(arr) - 1):
        for j in range(len(arr) - i - 1):
            if(arr[j] > arr[j +1]):
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp

def binary_search():
    low = 0
    high = len(arr) - 1

    while(high >= low):
        mid_value = low + (high - low) // 2

        try:
            if(target == arr[mid_value]):
                print(f"The targeted value of {target} found in index {mid_value}")
                found = True
                break

            elif(target < arr[mid_value]):
                high = mid_value - 1

            elif(target > arr[mid_value]):
                low = mid_value + 1
        except:
            print(f"Error while finding the targeted value")
            break

    if not found:
        print(f"Error while finding the targeted element in the array")

bubble_sort()
binary_search()