TEXT_TO_SQL_EXAMPLES = [
    # ========================================================
    # SELECT ALL
    # ========================================================

    {"question": "show all employees", "sql": "SELECT * FROM employees;"},
    {"question": "list all employees", "sql": "SELECT * FROM employees;"},
    {"question": "get every employee", "sql": "SELECT * FROM employees;"},
    {"question": "display all employees", "sql": "SELECT * FROM employees;"},
    {"question": "give me all employees", "sql": "SELECT * FROM employees;"},

    {"question": "show all products", "sql": "SELECT * FROM products;"},
    {"question": "list all products", "sql": "SELECT * FROM products;"},
    {"question": "get every product", "sql": "SELECT * FROM products;"},
    {"question": "display all products", "sql": "SELECT * FROM products;"},
    {"question": "give me all products", "sql": "SELECT * FROM products;"},

    {"question": "show all orders", "sql": "SELECT * FROM orders;"},
    {"question": "list all orders", "sql": "SELECT * FROM orders;"},
    {"question": "get every order", "sql": "SELECT * FROM orders;"},
    {"question": "display all orders", "sql": "SELECT * FROM orders;"},
    {"question": "give me all orders", "sql": "SELECT * FROM orders;"},

    # ========================================================
    # SELECT COLUMN
    # ========================================================

    {"question": "show employee names", "sql": "SELECT name FROM employees;"},
    {"question": "list employee names", "sql": "SELECT name FROM employees;"},
    {"question": "get the names of employees", "sql": "SELECT name FROM employees;"},

    {"question": "show employee salaries", "sql": "SELECT salary FROM employees;"},
    {"question": "list all salaries", "sql": "SELECT salary FROM employees;"},
    {"question": "get salaries of employees", "sql": "SELECT salary FROM employees;"},

    {"question": "show product prices", "sql": "SELECT price FROM products;"},
    {"question": "list product prices", "sql": "SELECT price FROM products;"},
    {"question": "get the prices of products", "sql": "SELECT price FROM products;"},

    {"question": "show order statuses", "sql": "SELECT status FROM orders;"},
    {"question": "list order statuses", "sql": "SELECT status FROM orders;"},
    {"question": "get the status of orders", "sql": "SELECT status FROM orders;"},

    # ========================================================
    # COUNT
    # ========================================================

    {"question": "how many employees are there", "sql": "SELECT COUNT(*) FROM employees;"},
    {"question": "count all employees", "sql": "SELECT COUNT(*) FROM employees;"},
    {"question": "give me the employee count", "sql": "SELECT COUNT(*) FROM employees;"},
    {"question": "number of employees", "sql": "SELECT COUNT(*) FROM employees;"},

    {"question": "how many products are there", "sql": "SELECT COUNT(*) FROM products;"},
    {"question": "count all products", "sql": "SELECT COUNT(*) FROM products;"},
    {"question": "give me the product count", "sql": "SELECT COUNT(*) FROM products;"},
    {"question": "number of products", "sql": "SELECT COUNT(*) FROM products;"},

    {"question": "how many orders are there", "sql": "SELECT COUNT(*) FROM orders;"},
    {"question": "count all orders", "sql": "SELECT COUNT(*) FROM orders;"},
    {"question": "give me the order count", "sql": "SELECT COUNT(*) FROM orders;"},
    {"question": "number of orders", "sql": "SELECT COUNT(*) FROM orders;"},

    # ========================================================
    # AVG
    # ========================================================

    {"question": "what is the average employee salary", "sql": "SELECT AVG(salary) FROM employees;"},
    {"question": "show the average salary", "sql": "SELECT AVG(salary) FROM employees;"},
    {"question": "calculate average salary", "sql": "SELECT AVG(salary) FROM employees;"},
    {"question": "find the mean salary", "sql": "SELECT AVG(salary) FROM employees;"},

    {"question": "what is the average product price", "sql": "SELECT AVG(price) FROM products;"},
    {"question": "show the average price", "sql": "SELECT AVG(price) FROM products;"},
    {"question": "calculate average product price", "sql": "SELECT AVG(price) FROM products;"},

    {"question": "what is the average order amount", "sql": "SELECT AVG(total_amount) FROM orders;"},
    {"question": "show average total order amount", "sql": "SELECT AVG(total_amount) FROM orders;"},
    {"question": "calculate average total amount", "sql": "SELECT AVG(total_amount) FROM orders;"},

    # ========================================================
    # DISTINCT
    # ========================================================

    {"question": "show distinct employee departments", "sql": "SELECT DISTINCT department FROM employees;"},
    {"question": "list unique departments", "sql": "SELECT DISTINCT department FROM employees;"},
    {"question": "show different departments", "sql": "SELECT DISTINCT department FROM employees;"},

    {"question": "show distinct product categories", "sql": "SELECT DISTINCT category FROM products;"},
    {"question": "list unique product categories", "sql": "SELECT DISTINCT category FROM products;"},
    {"question": "show different product categories", "sql": "SELECT DISTINCT category FROM products;"},

    {"question": "show distinct order statuses", "sql": "SELECT DISTINCT status FROM orders;"},
    {"question": "list unique order statuses", "sql": "SELECT DISTINCT status FROM orders;"},
    {"question": "show different order statuses", "sql": "SELECT DISTINCT status FROM orders;"},

    # ========================================================
    # WHERE >
    # ========================================================

    {"question": "show employees with salary greater than 60000", "sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "get employees earning more than 60000", "sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "list employees whose salary is above 60000", "sql": "SELECT * FROM employees WHERE salary > 60000;"},

    {"question": "show employees older than 30", "sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "get employees with age greater than 30", "sql": "SELECT * FROM employees WHERE age > 30;"},

    {"question": "show products priced above 50", "sql": "SELECT * FROM products WHERE price > 50;"},
    {"question": "get products where price is greater than 50", "sql": "SELECT * FROM products WHERE price > 50;"},

    {"question": "show orders above 100", "sql": "SELECT * FROM orders WHERE total_amount > 100;"},
    {"question": "get orders with total amount greater than 100", "sql": "SELECT * FROM orders WHERE total_amount > 100;"},

    # ========================================================
    # WHERE <
    # ========================================================

    {"question": "show employees younger than 30", "sql": "SELECT * FROM employees WHERE age < 30;"},
    {"question": "get employees with age less than 30", "sql": "SELECT * FROM employees WHERE age < 30;"},

    {"question": "show products cheaper than 50", "sql": "SELECT * FROM products WHERE price < 50;"},
    {"question": "get products where price is less than 50", "sql": "SELECT * FROM products WHERE price < 50;"},
    {"question": "list products priced below 50", "sql": "SELECT * FROM products WHERE price < 50;"},

    {"question": "show orders below 100", "sql": "SELECT * FROM orders WHERE total_amount < 100;"},
    {"question": "get orders with total amount less than 100", "sql": "SELECT * FROM orders WHERE total_amount < 100;"},

    # ========================================================
    # LIMIT
    # ========================================================

    {"question": "show top 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},
    {"question": "show first 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},
    {"question": "get 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},

    {"question": "show top 5 products", "sql": "SELECT * FROM products LIMIT 5;"},
    {"question": "show first 5 products", "sql": "SELECT * FROM products LIMIT 5;"},
    {"question": "get 5 products", "sql": "SELECT * FROM products LIMIT 5;"},

    {"question": "show top 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},
    {"question": "show first 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},
    {"question": "get 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},

    # ========================================================
    # ORDER BY DESC
    # ========================================================

    {"question": "show employees ordered by salary from highest", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},
    {"question": "sort employees by salary highest first", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},
    {"question": "order employees by salary descending", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},

    {"question": "show products ordered by price highest first", "sql": "SELECT * FROM products ORDER BY price DESC;"},
    {"question": "sort products by price descending", "sql": "SELECT * FROM products ORDER BY price DESC;"},

    {"question": "show orders ordered by total amount highest first", "sql": "SELECT * FROM orders ORDER BY total_amount DESC;"},
    {"question": "sort orders by total amount descending", "sql": "SELECT * FROM orders ORDER BY total_amount DESC;"},

    # ========================================================
    # ORDER BY ASC
    # ========================================================

    {"question": "show employees ordered by salary lowest first", "sql": "SELECT * FROM employees ORDER BY salary ASC;"},
    {"question": "sort employees by salary ascending", "sql": "SELECT * FROM employees ORDER BY salary ASC;"},

    {"question": "show products ordered by price lowest first", "sql": "SELECT * FROM products ORDER BY price ASC;"},
    {"question": "sort products by price ascending", "sql": "SELECT * FROM products ORDER BY price ASC;"},

    {"question": "show orders ordered by total amount lowest first", "sql": "SELECT * FROM orders ORDER BY total_amount ASC;"},
    {"question": "sort orders by total amount ascending", "sql": "SELECT * FROM orders ORDER BY total_amount ASC;"},
]


def build_text_to_sql_corpus(schema_text: str, prompt_builder) -> list[dict]:
    corpus = []

    for example in TEXT_TO_SQL_EXAMPLES:
        prompt = prompt_builder.build_inference_prompt(schema_text=schema_text, question=example["question"])
        corpus.append({"prompt": prompt, "question": example["question"], "sql": example["sql"]})

    return corpus