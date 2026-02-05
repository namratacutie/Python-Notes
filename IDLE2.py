size = 30
top_pointer = -1
stack = [None for i in range(size)]

def push(element):
    global top_pointer

    if top_pointer < size - 1:
        top_pointer += top_pointer
        stack[top_pointer] = element
        return True
    else:
        return False

push("Suprem")
push("Lawarna")

print(stack)
