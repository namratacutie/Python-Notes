# ========== CLASS DEFINITION ==========
# TreasureChest is a blueprint for creating question objects.
# Each object holds a question, its correct answer, and the max points.
class TreasureChest:

    # Constructor: called automatically when we create a new TreasureChest object
    # e.g. TreasureChest("3 * 2", "6", "10") creates one treasure chest
    # The double underscore (__) makes the attributes PRIVATE,
    # meaning they can only be accessed from inside the class via methods.
    def __init__(self, Pquestion, Panswer, points):
        self.__question = Pquestion
        self.__answer = Panswer
        self.__points = points

    # Returns the question string (e.g. "3 * 2") so we can display it to the user
    def getQuestion(self):
        return self.__question

    # Compares the user's answer with the stored correct answer
    # We convert self.__answer to int because it's stored as a string from the file
    def checkAnswer(self, Panswer):
        try:
            return int(self.__answer) == Panswer
        except (ValueError, TypeError):
            return False

    # Returns points based on how many attempts the user took:
    #   1st attempt = full points, 2nd = half, 3rd/4th = quarter, 5+ = 0
    def getPoints(self, attempts):
        try:
            points = int(self.__points)
            if attempts == 1:
                return points
            elif attempts == 2:
                return points // 2
            elif attempts == 3 or attempts == 4:
                return points // 4
            else:
                return 0
        except (ValueError, TypeError):
            return 0

# ========== READING DATA FROM FILE ==========
# This list will hold TreasureChest OBJECTS (not raw strings)
arrayTreasure = []

def readData():
    try:
        # Open the text file for reading
        with open("TreasureChestData.txt", "r") as file:
            # Each line looks like: "3 * 2, 6, 10"
            for line in file:
                # Split the line by comma into a list: ["3 * 2", " 6", " 10"]
                data = line.strip().split(",")
                if len(data) >= 3:
                    # Create a TreasureChest object using the 3 parts
                    chest = TreasureChest(data[0].strip(), data[1].strip(), data[2].strip())
                    # Add the OBJECT to the list (not the raw string)
                    arrayTreasure.append(chest)
    except FileNotFoundError:
        print("Error: 'TreasureChestData.txt' not found.")
    except Exception as e:
        print(f"An error occurred while reading data: {e}")

# Call readData() to load all questions into arrayTreasure
readData()

# ========== MAIN GAME LOOP ==========
def main():
    if not arrayTreasure:
        print("No treasure chests loaded. Check your data file.")
        return

    print("Welcome to the Treasure Chest Game!")
    
    while True:
        print(f"\nThere are {len(arrayTreasure)} treasure chests available.")
        user_input = input("Pick a treasure chest to open (1-5) or 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("Thanks for playing! Goodbye.")
            break
            
        try:
            choice = int(user_input)
            if 1 <= choice <= len(arrayTreasure):
                chest = arrayTreasure[choice - 1]
                result = False
                attempts = 0
                
                print(f"\nOpening Treasure Chest #{choice}...")
                
                # Keep asking until the user gets the right answer
                while not result:
                    try:
                        answer_input = input(f"Question: {chest.getQuestion()} = ")
                        if answer_input.strip().lower() == 'skip':
                            print("Skipping this chest.")
                            break
                        
                        answer = int(answer_input)
                        attempts += 1
                        result = chest.checkAnswer(answer)
                        
                        if result:
                            # Correct! Calculate and display points based on attempt count
                            score = chest.getPoints(attempts)
                            print(f"Correct! You scored {score} points in {attempts} attempt(s)!")
                        else:
                            print("Wrong answer! Hint: Try again (or type 'skip' to give up).")
                    except ValueError:
                        print("Invalid input. Please enter a whole number as your answer.")
            else:
                print(f"Invalid choice. Please pick a number between 1 and {len(arrayTreasure)}.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5, or 'q' to quit.")

if __name__ == "__main__":
    main()