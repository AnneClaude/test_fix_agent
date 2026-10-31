"""A simple bounded stack."""


class BoundedStack:
    """A last-in, first-out (LIFO) stack that holds at most `capacity` items."""

    def __init__(self, capacity):
        """Create an empty stack. Raise ValueError if capacity is not positive."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = []

    def push(self, item):
        """Add item to the top of the stack. Raise OverflowError if the stack is full."""
        if len(self._items) > self.capacity:
            raise OverflowError("stack is full")
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raise IndexError if the stack is empty."""
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop(0)

    def peek(self):
        """Return the top item without removing it. Raise IndexError if the stack is empty."""
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def __len__(self):
        """Return the number of items in the stack."""
        return len(self._items)

    def is_full(self):
        """Return True if the stack holds `capacity` items."""
        return len(self._items) == self.capacity