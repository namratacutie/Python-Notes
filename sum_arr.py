list = []
total = 0
n = int(input("Enter how many inputs you want to give? : "))

for i in range(n):
    num = int(input("Enter the values : "))
    list.append(num)
    total = total + num

print(f"The total sum is {total}")