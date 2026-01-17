fruits_list = ["Mango", "Banana", "Cherry", "Guava"]

for i in fruits_list:
    if "Apple" in fruits_list:
        fruits_list.remove("Apple")
    else:
        fruits_list.append("Apple")

print(fruits_list)
