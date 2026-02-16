# a) State one reason why the attributes would be declared as PRIVATE.
# Answer: To implement encapsulation, which prevents unauthorized or accidental modification of data from outside the class and ensures data integrity.

# b)
class Pet:

    def __init__(self, PetID, PetType, OwnerTelephone, DateRegistered, PetName, OwnerName):
        self.__PetID = PetID #Int
        self.__PetType = PetType #String
        self.__OwnerTelephone = OwnerTelephone #String
        self.__DateRegistered = DateRegistered #Date
        self.__PetName = PetName #String
        self.__OwnerName = OwnerName #String

    def SetPetID(self, PetID):
        self.__PetID = PetID

    def SetDateRegistered(self, Date):
        self.__DateRegistered = Date

    def GetPetName(self):
        return self.__PetName

    def GetOwnerTelephone(self):
        return self.__OwnerTelephone

    def GetOwnerName(self):
        return self.__OwnerName

if __name__ == "__main__":
    myPet1 = Pet("P001", "Cat", "+9779742246521", "2026-02-16", "Aayush", "Lawarna")
    
    # Verifying getters
    print(f"Pet Name: {myPet.GetPetName()}")
    print(f"Pet Owner : {myPet.GetOwnerName()}")
    print(f"Owner Telephone: {myPet.GetOwnerTelephone()}")
    
    # Verifying setters
    myPet.SetPetID("P002")
    myPet.SetDateRegistered("2024-05-21")       

    print(myPet)