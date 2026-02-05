class Train:
    def __init__(self, trainIDNumber, route):
        self.__trainIDNumber = trainIDNumber
        self.__route = route

    def GetTrainIDNumber(self):
        return self.__trainIDNumber

    def GetRoute(self):
        return self.__route

firstTrain = Train("12ADV", 132)
secondTrain = Train("33ART",20)
thirdTrain = Train("9FKF",3)
fourthTrain = Train("21VBC",24)

print(firstTrain.GetTrainIDNumber())
print(firstTrain.GetRoute())