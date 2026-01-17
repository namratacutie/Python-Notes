arr = [4, 0, 2, 8, 0, 6]
#Sort the array and keep the 0's at the end of the array without loosing the sequence of the non-zero elements

zero_arr = []
non_zero_arr = []
final_arr =  []

for i in range(len(arr)):
    if(arr[i] == 0 ):
        zero_arr.append(arr[i])

    elif(arr[i] != 0):
        non_zero_arr.append(arr[i])

final_arr.extend(non_zero_arr + zero_arr)


print("Initial List = ",arr)
print("List with zero elements = ", zero_arr)
print("List with non zero elements = ", non_zero_arr)
print("Final List = ", final_arr)