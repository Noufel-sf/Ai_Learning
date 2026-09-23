"""
=============================================================================
LESSON 01: PYTHON FUNDAMENTALS FOR SOFTWARE ENGINEERS
=============================================================================
This file is executable. Run it with:
    python 00_python_crash_course/01_basics_and_data_structures.py
=============================================================================
"""

def separator(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


# ---------------------------------------------------------------------------
# 1. CORE SYNTAX: NO CURLY BRACES, NO SEMICOLONS
# ---------------------------------------------------------------------------
separator("1. Core Syntax & Types")

# Dynamic, but STRONGLY typed (unlike JavaScript, "5" + 5 throws an error)
age = 28               # int
weight = 72.5          # float
name = "Noufel"        # str
is_engineer = True     # bool (Capitalized!)
nothing = None         # NoneType (Python's 'null' or 'nil')

# Formatted strings (f-strings) - similar to JS template literals `${...}`
print(f"Developer: {name} | Age: {age} | Weight: {weight}kg | Active: {is_engineer}")

# Type checking & annotations (hints for readability and IDEs)
def greet(user: str) -> str:
    return f"Hello, {user}!"

print(greet(name))


# ---------------------------------------------------------------------------
# 2. THE BIG 4 DATA STRUCTURES (CRITICAL FOR ML)
# ---------------------------------------------------------------------------
separator("2. The Big 4 Data Structures")

# A. LIST (Dynamic Array - like JS Array or Java ArrayList)
fruits = ["apple", "banana", "cherry", "date"]
fruits.append("elderberry")

# Slicing: list[start : end : step] (End index is EXCLUSIVE)
# In ML, you will slice matrices and vectors constantly!
print("First item:", fruits[0])
print("Last item (negative indexing):", fruits[-1])
print("Items 1 to 3:", fruits[1:4])
print("Reversed list:", fruits[::-1])

# B. TUPLE (Immutable sequence - once created, cannot be modified)
# Perfect for fixed data like coordinates, model shapes: (batch_size, channels, height, width)
coordinates = (1920, 1080)
# Tuple unpacking (destructuring):
width, height = coordinates
print(f"Screen resolution: {width}x{height}")

# C. DICTIONARY (Hash map - key/value pairs, like JS object or Java HashMap)
model_config = {
    "model_name": "linear_regression",
    "learning_rate": 0.01,
    "epochs": 100,
}
# Accessing & updating
print("Learning rate:", model_config["learning_rate"])
model_config["optimizer"] = "adam"
print("Config keys:", list(model_config.keys()))

# Safe access with .get() (won't throw KeyError if missing)
batch_size = model_config.get("batch_size", 32)  # default value 32
print("Batch size (defaulted):", batch_size)

# D. SET (Unordered collection of unique items)
raw_labels = ["cat", "dog", "cat", "bird", "dog"]
unique_labels = set(raw_labels)
print("Unique labels:", unique_labels)


# ---------------------------------------------------------------------------
# 3. LIST & DICT COMPREHENSIONS (PYTHON'S .map() and .filter())
# ---------------------------------------------------------------------------
separator("3. Comprehensions (Clean Data Transformation)")

# In JS: numbers.filter(x => x > 0).map(x => x ** 2)
# In Python:
numbers = [-5, -2, 0, 3, 7, 10]

# List comprehension: [EXPRESSION for ITEM in ITERABLE if CONDITION]
squared_positives = [x ** 2 for x in numbers if x > 0]
print("Squared positives:", squared_positives)

# Dict comprehension: create a lookup map
labels = ["cat", "dog", "bird"]
label_to_id = {label: idx for idx, label in enumerate(labels)}
print("Label to ID mapping:", label_to_id)


# ---------------------------------------------------------------------------
# 4. FUNCTIONS: *args, **kwargs, AND LAMBDAS
# ---------------------------------------------------------------------------
separator("4. Functions, *args, **kwargs")

# Default arguments
def train_model(epochs: int = 10, lr: float = 0.001):
    return f"Trained for {epochs} epochs with lr={lr}"

print(train_model())
print(train_model(epochs=50, lr=0.05))  # Named keyword arguments!

# *args (variable positional args) and **kwargs (variable keyword args)
# You will see **kwargs heavily in PyTorch and Scikit-Learn configs
def print_hyperparameters(*args, **kwargs):
    print("Positional args:", args)
    print("Keyword args (kwargs):", kwargs)

print_hyperparameters("adam", "cross_entropy", epochs=20, lr=0.001, dropout=0.2)

# Lambda functions (anonymous inline functions, like (x) => x * 2 in JS)
square = lambda x: x ** 2
print("Lambda square of 6:", square(6))


# ---------------------------------------------------------------------------
# 5. MINI CHALLENGE FOR YOU TO TRY
# ---------------------------------------------------------------------------
separator("5. Mini Exercise")
print("""
Check out the challenge below in this file. Try completing it!
""")

def normalize_scores(scores: list[float]) -> list[float]:
    """
    ML Preprocessing Exercise:
    Given a list of scores, find the min and max,
    and return a new list with Min-Max normalized values between 0.0 and 1.0.
    Formula: (x - min) / (max - min)
    
    Try writing it using min(), max(), and a list comprehension!
    """
    # TODO: Implement this!
    pass


if __name__ == "__main__":
    test_scores = [10.0, 25.0, 50.0, 75.0, 100.0]
    print("Input scores:", test_scores)
    # result = normalize_scores(test_scores)
    # print("Normalized:", result)
    # Expected output: [0.0, 0.2, 0.5333..., 0.8666..., 1.0]
