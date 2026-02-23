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
    try:
        with open("Pictures.txt", "r") as file:
            lines = file.readlines()

            for i in range(0, len(lines), 4):
                if i + 3 < len(lines):
                    description = lines[i].strip()
                    width = lines[i+1].strip()
                    height = lines[i + 2].strip()
                    colour = lines[i + 3].strip()

                    Pic = Picture(description, width, height, colour)

                    PictureArr.append(Pic)

    except FileNotFoundError:
        print("Error: 'Pictures.txt' not found.")
    except Exception as e:
        print(f"An error occurred while reading data: {e}")

def main():
    ReadData()


if __name__ == "__main__":
    main()