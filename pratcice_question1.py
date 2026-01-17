#Create three python functions for any maths operation of your choice the functions must return their values. Now create a function name final_answer and print all the result there

def user_input():
    first_num = int(input("Enter the first number : "))
    second_num = int(input("Enter the second number : "))
    return first_num, second_num

def add(first_num, second_num):
    return first_num + second_num

def sub(first_num, second_num):
    return first_num - second_num

def mul(first_num, second_num):
    return first_num * second_num

def div(first_num, second_num):
    return first_num / second_num

def final_answer():
    first_num, second_num = user_input()
    
    print(f"\nThe sum of {first_num} and {second_num} is {add(first_num, second_num)}\n")
    print(f"The difference of {first_num} and {second_num} is {sub(first_num, second_num)}\n")
    print(f"The product of {first_num} and {second_num} is {mul(first_num, second_num)}\n")
    print(f"The quotient of {first_num} and {second_num} is {div(first_num, second_num)}\n")

final_answer()