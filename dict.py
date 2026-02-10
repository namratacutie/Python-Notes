stock = {
    "apples" : 5,
    "bananas" : 12,
}

stock.update({"apples" : 10, "orange" : 8})

print(stock)

student_profile = {}

student_profile.update({"name" : "Marcus", "age" : 17, "subject" : ["Computer Science", "Maths", "Physics"]})

fruit_prices = {
    "apple" : 0.50,
    "banana" : 0.30,
    "cherry" : 0.85,
}

print(fruit_prices.get("apple"))

print(fruit_prices.get("mango", "0.0"))

products = {}

for i in range(3):
    productName = input("Enter the product name : ")
    productPrice = int(input("Enter the product Price : "))
    print("\n")
    products[productName] = productPrice
    

print(products)