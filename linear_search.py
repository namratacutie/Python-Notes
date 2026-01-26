arr = [1,2,3,4,5,6]
target = 10
found = False

for i in range(len(arr)):
    if arr[i] == target:
        print(f"The target value {target} is in index {i}")
        found = True
        break

if not found:
    print("Not found")