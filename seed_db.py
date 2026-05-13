"""
Seeds a small SQLite 'sales' database for the Text2SQL demo.
Idempotent — safe to run multiple times.
"""
import os
import sqlite3
from datetime import date, timedelta
import random

from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DATABASE_PATH", "demo.db")

SCHEMA_SQL = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS employees;

CREATE TABLE customers (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    country      TEXT NOT NULL,
    signup_date  DATE NOT NULL
);

CREATE TABLE employees (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL,
    region       TEXT NOT NULL,
    hired_on     DATE NOT NULL
);

CREATE TABLE products (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    unit_price   REAL NOT NULL,
    in_stock     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    employee_id  INTEGER NOT NULL REFERENCES employees(id),
    order_date   DATE NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending','shipped','delivered','cancelled')),
    total        REAL NOT NULL
);

CREATE TABLE order_items (
    id           INTEGER PRIMARY KEY,
    order_id     INTEGER NOT NULL REFERENCES orders(id),
    product_id   INTEGER NOT NULL REFERENCES products(id),
    quantity     INTEGER NOT NULL,
    unit_price   REAL NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
"""


def seed():
    # Remove old DB if it exists, to keep things deterministic
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)

    random.seed(42)

    # Customers
    countries = ["US", "UK", "DE", "FR", "JP", "BR", "IN", "CA", "AU", "MX"]
    customer_names = [
        "Alice Chen", "Bob Patel", "Carlos Mendez", "Diana Schmidt", "Ethan Wright",
        "Fiona O'Connor", "Gabriel Silva", "Hana Tanaka", "Isaac Cohen", "Julia Dubois",
        "Kenji Yamamoto", "Lara Ivanova", "Mateo Rossi", "Nadia Khan", "Oliver Brown",
        "Priya Sharma", "Quinn Murphy", "Rafael Costa", "Sofia Garcia", "Tomas Novak",
        "Uma Reddy", "Victor Hugo", "Wendy Park", "Xavier Lopez", "Yara Kim",
    ]
    customers = []
    for i, name in enumerate(customer_names, start=1):
        email = name.lower().replace(" ", ".").replace("'", "") + "@example.com"
        country = random.choice(countries)
        signup = date(2023, 1, 1) + timedelta(days=random.randint(0, 700))
        customers.append((i, name, email, country, signup.isoformat()))
    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)", customers
    )

    # Employees
    employees_data = [
        (1, "Sam Reed",      "Sales Rep",      "North America", "2022-03-15"),
        (2, "Lin Zhao",      "Sales Rep",      "Asia Pacific",  "2021-07-20"),
        (3, "Marco Rossi",   "Sales Manager",  "Europe",        "2020-01-10"),
        (4, "Aisha Bello",   "Sales Rep",      "Europe",        "2023-02-05"),
        (5, "Jin Park",      "Sales Rep",      "Asia Pacific",  "2022-11-12"),
    ]
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees_data)

    # Products
    products_data = [
        (1,  "Wireless Mouse",        "Electronics", 24.99, 1),
        (2,  "Mechanical Keyboard",   "Electronics", 89.50, 1),
        (3,  "USB-C Hub",             "Electronics", 39.00, 1),
        (4,  "Standing Desk",         "Furniture",   349.00, 1),
        (5,  "Office Chair",          "Furniture",   249.99, 1),
        (6,  "Desk Lamp",             "Furniture",   45.00, 1),
        (7,  "Notebook (3-pack)",     "Stationery",  12.50, 1),
        (8,  "Pen Set",               "Stationery",  8.75,  1),
        (9,  "Whiteboard",            "Stationery",  65.00, 0),
        (10, "Coffee Mug",            "Kitchen",     14.00, 1),
        (11, "Water Bottle",          "Kitchen",     19.99, 1),
        (12, "Espresso Machine",      "Kitchen",     299.00, 1),
        (13, "Backpack",              "Bags",        59.00, 1),
        (14, "Laptop Sleeve",         "Bags",        29.99, 1),
        (15, "Travel Suitcase",       "Bags",        179.00, 0),
    ]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products_data)

    # Orders + order items
    statuses = ["pending", "shipped", "delivered", "delivered", "delivered", "cancelled"]
    order_id = 1
    item_id = 1
    for _ in range(300):
        cust_id = random.randint(1, len(customer_names))
        emp_id = random.randint(1, len(employees_data))
        order_date = date(2024, 1, 1) + timedelta(days=random.randint(0, 500))
        status = random.choice(statuses)

        # 1–4 items per order
        line_items = []
        for _ in range(random.randint(1, 4)):
            prod = random.choice(products_data)
            qty = random.randint(1, 5)
            line_items.append((prod[0], qty, prod[3]))
        total = round(sum(qty * price for _, qty, price in line_items), 2)

        cur.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?)",
            (order_id, cust_id, emp_id, order_date.isoformat(), status, total),
        )
        for prod_id, qty, price in line_items:
            cur.execute(
                "INSERT INTO order_items VALUES (?,?,?,?,?)",
                (item_id, order_id, prod_id, qty, price),
            )
            item_id += 1
        order_id += 1

    conn.commit()

    # Quick sanity check
    print(f"Seeded {DB_PATH}")
    for table in ("customers", "employees", "products", "orders", "order_items"):
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")
    conn.close()


if __name__ == "__main__":
    seed()
