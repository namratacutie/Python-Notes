# Stack follows LIFO

stack_len = 30
stack = [None for i in range(stack_len)]
top_pointer = -1 

def push(element):
    global top_pointer

    try:
        if(top_pointer < stack_len - 1):
            top_pointer += 1
            stack[top_pointer] = element

        else:
            print(f"Stack Overflowed")
    except:
        print(f"Error while pushing elements to the array")

def pop():
    global top_pointer

    if(top_pointer == -1):
        print(f"Stack Underflowed")
        return None

    else:
        element = stack[top_pointer]
        top_pointer -= 1
        return element

    

push(1)
push(2)
push(3)
push(4)

print(pop())
print(stack)
