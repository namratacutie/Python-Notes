#Making a new empty list 
my_list = []

#Printing the list 
print(my_list)

#Making a list with elements
my_second_list = ["Lawarna", "Pra", 2]

#Printing the list 
print(my_second_list)

#Accessing the list elements
my_element_list = ["Islington", "Herald", "Kavya"]
print(my_element_list[0], my_element_list[2])

#Accessing multi-dimensional list and accessing it 
my_multi_list = [["User1"], ["Lawarna", "Aree", "+977 9742246521"]]
print(my_multi_list[1][2])

#Accessing the element from negative indexing
my_multi_list = [["User1"], ["Lawarna", "Aree", "+977 9742246521"]]
print(my_multi_list[-1][-2])

#Finding the length of the list
my_list = ["Lawarna", "Pra", 2]
print(len(my_list))

#Adding the element to the list uisng append
my_list = ["Lawarna", "Pra", 2]
my_list.append("Aree")
print(my_list)

#Adding the element to the list uisng insert
my_list = ["Lawarna", "Pra", 2]
my_list.insert(1, "Aree")
print(my_list)

#Counting the number of occurance of any element in the list 
my_list = ["Lawarna", "Pra", 2]
print(my_list.count("Pra")) 

#Removing the element from the list using del keyword
my_list = ["Lawarna", "Pra", 2]
del my_list[1]
print(my_list)

#Removing the element from the list using remove method
my_list = ["Lawarna", "Pra", 2]
my_list.remove("Pra")
print(my_list)

#Removing the element from the list using pop method
my_list = ["Lawarna", "Pra", 2]
my_list.pop(1)
print(my_list)

#Clearing the list
my_list = ["Lawarna", "Pra", 2]
my_list.clear()
print(my_list)

#Deleting the list
my_list = ["Lawarna", "Pra", 2]
del my_list
print(my_list)

#Copying the list
my_list = ["Lawarna", "Pra", 2]
my_list.copy()
print(my_list)

#Slicing the list
my_list = ["Lawarna", "Pra", 2]
print(my_list[0:2])

#Sorting the list
my_list = ["Lawarna", "Pra", 2]
my_list.sort()
print(my_list)

#Reversing the list
my_list = ["Lawarna", "Pra", 2]
my_list.reverse()
print(my_list)

#Reversing the list using slice
my_list = ["Lawarna", "Pra", 2]
my_list[::-1]
print(my_list)

#Reversing the list using reverse method
my_list = ["Lawarna", "Pra", 2]
my_list.reverse()
print(my_list)

#Reversing the list using slice
my_list = ["Lawarna", "Pra", 2]
my_list[::-1]
print(my_list)

#Reversing the list using reverse method
my_list = ["Lawarna", "Pra", 2]
my_list.reverse()
print(my_list)

#Reversing the list using slice
my_list = ["Lawarna", "Pra", 2]
my_list[::-1]
print(my_list)

