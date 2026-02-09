#r - Opens a file in, “Read” mode
# w - Opens a file in, “Write” mode
# a - Opens a file in, “Append” mode
# r+ - Opens a file in, “Read and Write” mode

#Opening the file
# Readaing a file ---> Can't write while open with read method and need to close the file
file = open("example.txt", "r")
content = file.read()
print(content)
file.close()

#With (with open) method you can write the data and does not need to close the file
#reading a file
with open("example.txt", "r") as file:
    content = file.read() # File is automatically closed here!

#From a file to Array
# Assuming 'students.txt' contains names on each line
student_list = []

with open("students.txt", "r") as file:
    for line in file:
        student_list.append(line.strip())

print(student_list)

#From array to file

new_students = ["Regan", "Ujain", "Lakshya"]

# We use 'a' to add to the file without deleting previous names
with open("students.txt", "a") as file:
    for i in new_students:
        file.write(i + "\n") # Adding "\n" ensures each name is on a new line


# strip(): Removes invisible clutter

# When we read a line from a file, it usually includes a hidden newline character (\n) at the end. strip() removes whitespace

# and newlines from the start and end of a string.


# split(): Break things apart:

# split() turns a single string into a list based on a "separator" (like a comma or a space). This is essential for reading CSV
    
# (Comma Separated Values) files.