SELECT_ALL_EXAMPLES = [

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
]

SELECT_COLUMN_EXAMPLES = [
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
    ]


COUNT_EXAMPLES = [
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
    {"question": "number of orders", "sql": "SELECT COUNT(*) FROM orders;"},]

AVG_EXAMPLES = [
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

    # Extra AVG - employees
    {"question": "what is the average salary of employees", "sql": "SELECT AVG(salary) FROM employees;"},
    {"question": "calculate the mean employee salary", "sql": "SELECT AVG(salary) FROM employees;"},

    # Extra AVG - products
    {"question": "what is the mean price of products", "sql": "SELECT AVG(price) FROM products;"},
    {"question": "find the average price of all products", "sql": "SELECT AVG(price) FROM products;"},

    # Extra AVG - orders
    {"question": "what is the mean order amount", "sql": "SELECT AVG(total_amount) FROM orders;"},
    {"question": "find the average amount of all orders", "sql": "SELECT AVG(total_amount) FROM orders;"},

    {"question": "what is the average order quantity","sql": "SELECT AVG(quantity) FROM orders;"},

    {"question": "find the average quantity of orders","sql": "SELECT AVG(quantity) FROM orders;"}]

DISTINCT_EXAMPLES = [
    {"question": "show distinct employee departments", "sql": "SELECT DISTINCT department FROM employees;"},
    {"question": "list unique departments", "sql": "SELECT DISTINCT department FROM employees;"},
    {"question": "show different departments", "sql": "SELECT DISTINCT department FROM employees;"},

    {"question": "show distinct product categories", "sql": "SELECT DISTINCT category FROM products;"},
    {"question": "list unique product categories", "sql": "SELECT DISTINCT category FROM products;"},
    {"question": "show different product categories", "sql": "SELECT DISTINCT category FROM products;"},

    {"question": "show distinct order statuses", "sql": "SELECT DISTINCT status FROM orders;"},
    {"question": "list unique order statuses", "sql": "SELECT DISTINCT status FROM orders;"},
    {"question": "show different order statuses", "sql": "SELECT DISTINCT status FROM orders;"}]

WHERE_GREATER_THAN_EXAMPLES = [
    {"question": "show employees with salary greater than 60000","sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "get employees earning more than 60000","sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "show employees older than 30","sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "list employees above age 30","sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "show products priced above 50","sql": "SELECT * FROM products WHERE price > 50;"},
    {"question": "find products costing more than 50","sql": "SELECT * FROM products WHERE price > 50;"},
    {"question": "show orders above 100","sql": "SELECT * FROM orders WHERE total_amount > 100;"},
    {"question": "find orders worth more than 100","sql": "SELECT * FROM orders WHERE total_amount > 100;"},
    {"question": "find employees above age 30","sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "show products costing more than 50","sql": "SELECT * FROM products WHERE price > 50;"},
    {"question": "find orders worth more than 100","sql": "SELECT * FROM orders WHERE total_amount > 100;"},
    {"question": "list employees with salary above 60000","sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "show orders where quantity is greater than 2", "sql": "SELECT * FROM orders WHERE quantity > 2;"},
    {"question": "find orders with quantity greater than 2", "sql": "SELECT * FROM orders WHERE quantity > 2;"},
]

WHERE_LESS_THAN_EXAMPLES = [
    # Employees
    {"question": "show employees younger than 30","sql": "SELECT * FROM employees WHERE age < 30;"},
    {"question": "get employees with age less than 30","sql": "SELECT * FROM employees WHERE age < 30;"},
    {"question": "show employees under age 30","sql": "SELECT * FROM employees WHERE age < 30;"},
    {"question": "list employees whose age is below 30","sql": "SELECT * FROM employees WHERE age < 30;"},

    # Products
    {"question": "show products cheaper than 50","sql": "SELECT * FROM products WHERE price < 50;"},
    {"question": "get products where price is less than 50","sql": "SELECT * FROM products WHERE price < 50;"},
    {"question": "list products priced below 50","sql": "SELECT * FROM products WHERE price < 50;"},
    {"question": "show products costing less than 50","sql": "SELECT * FROM products WHERE price < 50;"},

    # Orders
    {"question": "show orders below 100","sql": "SELECT * FROM orders WHERE total_amount < 100;"},
    {"question": "get orders with total amount less than 100","sql": "SELECT * FROM orders WHERE total_amount < 100;"},
    {"question": "list orders with amount below 100","sql": "SELECT * FROM orders WHERE total_amount < 100;"},
    {"question": "show orders worth under 100","sql": "SELECT * FROM orders WHERE total_amount < 100;"},

    {"question": "show orders where quantity is less than 2", "sql": "SELECT * FROM orders WHERE quantity < 2;"},
    {"question": "find orders with quantity less than 2", "sql": "SELECT * FROM orders WHERE quantity < 2;"},
]

LIMIT_EXAMPLES = [
    {"question": "show top 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},
    {"question": "show first 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},
    {"question": "get 5 employees", "sql": "SELECT * FROM employees LIMIT 5;"},

    {"question": "show top 5 products", "sql": "SELECT * FROM products LIMIT 5;"},
    {"question": "show first 5 products", "sql": "SELECT * FROM products LIMIT 5;"},
    {"question": "get 5 products", "sql": "SELECT * FROM products LIMIT 5;"},

    {"question": "show top 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},
    {"question": "show first 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},
    {"question": "get 5 orders", "sql": "SELECT * FROM orders LIMIT 5;"},]

ORDER_BY_DESC_EXAMPLES = [
    {"question": "show employees ordered by salary from highest", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},
    {"question": "sort employees by salary highest first", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},
    {"question": "order employees by salary descending", "sql": "SELECT * FROM employees ORDER BY salary DESC;"},

    {"question": "show products ordered by price highest first", "sql": "SELECT * FROM products ORDER BY price DESC;"},
    {"question": "sort products by price descending", "sql": "SELECT * FROM products ORDER BY price DESC;"},

    {"question": "show orders ordered by total amount highest first", "sql": "SELECT * FROM orders ORDER BY total_amount DESC;"},
    {"question": "sort orders by total amount descending", "sql": "SELECT * FROM orders ORDER BY total_amount DESC;"},

    {"question": "show orders ordered by quantity highest first","sql": "SELECT * FROM orders ORDER BY quantity DESC;"},
    {"question": "sort orders by quantity descending","sql": "SELECT * FROM orders ORDER BY quantity DESC;"},
    # {"question": "sort employees from high salary to low salary","sql": "SELECT * FROM employees ORDER BY salary DESC;"},
    # {"question": "sort products from higher price to lower price","sql": "SELECT * FROM products ORDER BY price DESC;"},
    # {"question": "sort orders from higher quantity to lower quantity","sql": "SELECT * FROM orders ORDER BY quantity DESC;"},
    ]


ORDER_BY_ASC_EXAMPLES = [
    {"question": "show employees ordered by salary lowest first", "sql": "SELECT * FROM employees ORDER BY salary ASC;"},
    {"question": "sort employees by salary ascending", "sql": "SELECT * FROM employees ORDER BY salary ASC;"},

    {"question": "show products ordered by price lowest first", "sql": "SELECT * FROM products ORDER BY price ASC;"},
    {"question": "sort products by price ascending", "sql": "SELECT * FROM products ORDER BY price ASC;"},

    {"question": "show orders ordered by total amount lowest first", "sql": "SELECT * FROM orders ORDER BY total_amount ASC;"},
    {"question": "sort orders by total amount ascending", "sql": "SELECT * FROM orders ORDER BY total_amount ASC;"},

    {"question": "show orders ordered by quantity lowest first","sql": "SELECT * FROM orders ORDER BY quantity ASC;"},
    {"question": "sort orders by quantity ascending","sql": "SELECT * FROM orders ORDER BY quantity ASC;"},
    # {"question": "sort employees from low salary to high salary","sql": "SELECT * FROM employees ORDER BY salary ASC;"},
    # {"question": "order employees from least paid to most paid","sql": "SELECT * FROM employees ORDER BY salary ASC;"},
    # {"question": "sort products from lower price to higher price","sql": "SELECT * FROM products ORDER BY price ASC;"},
    # {"question": "order products from least costly to most costly","sql": "SELECT * FROM products ORDER BY price ASC;"},
    # {"question": "sort orders from lower quantity to higher quantity","sql": "SELECT * FROM orders ORDER BY quantity ASC;"},
    # {"question": "order orders from fewer items to more items","sql": "SELECT * FROM orders ORDER BY quantity ASC;"},
]
WHERE_EQUALS_EXAMPLES = [
    {"question": "show employees where age equals 30",
     "sql": "SELECT * FROM employees WHERE age = 30;"},

    {"question": "show employees with age equal to 30",
     "sql": "SELECT * FROM employees WHERE age = 30;"},

    {"question": "show products where price equals 50",
     "sql": "SELECT * FROM products WHERE price = 50;"},

    {"question": "show products with price equal to 50",
     "sql": "SELECT * FROM products WHERE price = 50;"},

    {"question": "show orders where quantity equals 2",
     "sql": "SELECT * FROM orders WHERE quantity = 2;"},

    {"question": "show orders with quantity equal to 2",
     "sql": "SELECT * FROM orders WHERE quantity = 2;"},

    {"question": "employees aged exactly 27",
     "sql": "SELECT * FROM employees WHERE age = 27;"},

    {"question": "employees exactly 35 years old",
     "sql": "SELECT * FROM employees WHERE age = 35;"},

    {"question": "products costing exactly 40",
     "sql": "SELECT * FROM products WHERE price = 40;"},

    {"question": "products priced exactly 75",
     "sql": "SELECT * FROM products WHERE price = 75;"},

    {"question": "orders with exactly 3 items",
     "sql": "SELECT * FROM orders WHERE quantity = 3;"},

    {"question": "orders containing exactly 4 items",
     "sql": "SELECT * FROM orders WHERE quantity = 4;"},
]

WHERE_IN_EXAMPLES = [
    {
        "question": "show employees aged 25 or 30",
        "sql": "SELECT * FROM employees WHERE age IN (25, 30);"
    },
    {
        "question": "find employees whose age is 30 or 35",
        "sql": "SELECT * FROM employees WHERE age IN (30, 35);"
    },

    {
        "question": "show products priced 50 or 100",
        "sql": "SELECT * FROM products WHERE price IN (50, 100);"
    },
    {
        "question": "find products with price 25 or 50",
        "sql": "SELECT * FROM products WHERE price IN (25, 50);"
    },

    {
        "question": "show orders with quantity 1 or 2",
        "sql": "SELECT * FROM orders WHERE quantity IN (1, 2);"
    },
    {
        "question": "find orders having quantity 2 or 3",
        "sql": "SELECT * FROM orders WHERE quantity IN (2, 3);"
    },
]

WHERE_IN_EXAMPLES += [
    # employees
    {
        "question": "show employees who are either 24 or 32 years old",
        "sql": "SELECT * FROM employees WHERE age IN (24, 32);"
    },
    {
        "question": "find employees with age 27 or 34",
        "sql": "SELECT * FROM employees WHERE age IN (27, 34);"
    },
    {
        "question": "list employees whose age is one of 29 or 36",
        "sql": "SELECT * FROM employees WHERE age IN (29, 36);"
    },
    {
        "question": "get employees aged 22 or 40",
        "sql": "SELECT * FROM employees WHERE age IN (22, 40);"
    },

    # products
    {
        "question": "show products costing either 30 or 80",
        "sql": "SELECT * FROM products WHERE price IN (30, 80);"
    },
    {
        "question": "find products with price 40 or 90",
        "sql": "SELECT * FROM products WHERE price IN (40, 90);"
    },
    {
        "question": "list products whose price is one of 60 or 150",
        "sql": "SELECT * FROM products WHERE price IN (60, 150);"
    },
    {
        "question": "get products priced 35 or 75",
        "sql": "SELECT * FROM products WHERE price IN (35, 75);"
    },

    # orders
    {
        "question": "show orders containing either 2 or 4 items",
        "sql": "SELECT * FROM orders WHERE quantity IN (2, 4);"
    },
    {
        "question": "find orders with quantity 3 or 5",
        "sql": "SELECT * FROM orders WHERE quantity IN (3, 5);"
    },
    {
        "question": "list orders whose quantity is one of 4 or 6",
        "sql": "SELECT * FROM orders WHERE quantity IN (4, 6);"
    },
    {
        "question": "get orders having quantity 2 or 7",
        "sql": "SELECT * FROM orders WHERE quantity IN (2, 7);"
    },
]

WHERE_IN_EXAMPLES += [
    {
        "question": "employees who are 23 or 33 years old",
        "sql": "SELECT * FROM employees WHERE age IN (23, 33);"
    },
    {
        "question": "employees whose age is either 28 or 37",
        "sql": "SELECT * FROM employees WHERE age IN (28, 37);"
    },

    {
        "question": "products with a price of 55 or 110",
        "sql": "SELECT * FROM products WHERE price IN (55, 110);"
    },
    {
        "question": "products priced at either 65 or 140",
        "sql": "SELECT * FROM products WHERE price IN (65, 140);"
    },

    {
        "question": "orders having either 3 or 6 as quantity",
        "sql": "SELECT * FROM orders WHERE quantity IN (3, 6);"
    },
    {
        "question": "orders with quantity either 4 or 8",
        "sql": "SELECT * FROM orders WHERE quantity IN (4, 8);"
    },
]

TEXT_TO_SQL_EXAMPLES = (
    SELECT_ALL_EXAMPLES
    + SELECT_COLUMN_EXAMPLES
    + COUNT_EXAMPLES
    + AVG_EXAMPLES
    + DISTINCT_EXAMPLES
    + WHERE_GREATER_THAN_EXAMPLES
    + WHERE_LESS_THAN_EXAMPLES
    + WHERE_EQUALS_EXAMPLES
    + LIMIT_EXAMPLES
    + ORDER_BY_DESC_EXAMPLES
    + ORDER_BY_ASC_EXAMPLES
    + WHERE_IN_EXAMPLES
)

def build_text_to_sql_corpus(schema_text: str, prompt_builder) -> list[dict]:
    corpus = []

    for example in TEXT_TO_SQL_EXAMPLES:
        # prompt = prompt_builder.build_inference_prompt(schema_text=schema_text, question=example["question"])
        prompt = prompt_builder.build_question_prompt(example["question"])
        corpus.append({"prompt": prompt, "question": example["question"], "sql": example["sql"]})

    return corpus

print("Custom examples:", len(TEXT_TO_SQL_EXAMPLES))