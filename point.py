class Point():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(f"{self.x},{self.y}")

    def __repr__(self):
        return f"({self.x}, {self.y})"

    @property
    def is_valid(self):
        return self.x >= 0 and self.x <= 3 and self.y >= 0 and self.y <= 4

    @property
    def is_base(self):
        return self.x == -1 and (self.y == -1 or self.y == 5)