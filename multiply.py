def multiply(a, b):
    return a * b


if __name__ == "__main__":
    x = float(input("Enter first number: "))
    y = float(input("Enter second number: "))
    print(f"{x} * {y} = {multiply(x, y)}")
