class Picture:
    def __init__(self, Description, Width, Height, FrameColour):
        self.__Description = Description #STRING
        self.__Width = Width #INTEGER
        self.__Height = Height #INTEGER
        self.__FrameColour = FrameColour #STRING

    def GetDescription(self):
        return self.__Description

    def GetHeight(self):
        return self.__Height

    def GetWidth(self):
        return self.__Width

    def GetColour(self):
        return self.__FrameColour

    def SetDescription(self, Description):
        self.__Description = Description

PictureArr = [None for i in range(100)]

def ReadData():
    count = 0

    try:
        with open("Pictures.txt", "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    description = lines[i]
                    width = int(lines[i+1])
                    height = int(lines[i+2])
                    colour = lines[i+3]

                    Pic = Picture(description, width, height, colour)

                    if count < 100:
                        PictureArr[count] = Pic
                        count += 1
        return count

    except FileNotFoundError:
        print("Error: 'Pictures.txt' not found.")
    except Exception as e:
        print(f"An error occurred while reading data: {e}")

def main():
    count = ReadData()
    if count is None:
        count = 0
    
    print("\nEnter requirements for a picture:" + "\n")
    input_colour = input("Enter frame colour: ").strip().lower()
    input_width = input("Enter maximum width: ")
    input_height = input("Enter maximum height: ")
    
    try:
        input_width = int(input_width)
        input_height = int(input_height)

    except ValueError:
        print("Invalid width or height input.")
        return
    
    print("\nMatching pictures:")
    found = False
    for i in range(count):
        pic = PictureArr[i]
        if pic.GetColour().lower() == input_colour:
            if pic.GetWidth() <= input_width and pic.GetHeight() <= input_height:
                print(f"Description: {pic.GetDescription()}")
                print(f"Width: {pic.GetWidth()}")
                print(f"Height: {pic.GetHeight()}")
                print("-" * 25)
                found = True
    
    if not found:
        print("No matches found.")


if __name__ == "__main__":
    main()