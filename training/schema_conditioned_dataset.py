from __future__ import annotations
import random

LEGACY_SCHEMA={
    "employees":["id","name","department","salary","age","years_experience"],
    "products":["id","name","category","price","stock","rating"],
    "orders":["id","product_id","employee_id","quantity","total_amount","status"],
}

TRAIN_DATABASES=[
{
"schema":{
"employees":["id","name","department","salary","age","years_experience"],
"products":["id","name","category","price","stock","rating"],
"orders":["id","product_id","employee_id","quantity","total_amount","status"],
},
"profiles":{
"employees":{"entity":"employees","name":"name","category":"department","pay":"salary","age":"age","experience":"years_experience"},
"products":{"entity":"products","name":"name","category":"category","price":"price","stock":"stock","rating":"rating"},
"orders":{"entity":"orders","quantity":"quantity","amount":"total_amount","category":"status"},
}},
{
"schema":{
"developers":["developer_id","full_name","team","annual_pay","age_years","professional_experience"],
"inventory_items":["item_id","title","item_group","unit_cost","units_available","quality_score"],
"purchases":["purchase_id","item_id","developer_id","units","purchase_amount","state"],
},
"profiles":{
"developers":{"entity":"developers","name":"full_name","category":"team","pay":"annual_pay","age":"age_years","experience":"professional_experience"},
"inventory_items":{"entity":"inventory items","name":"title","category":"item_group","price":"unit_cost","stock":"units_available","rating":"quality_score"},
"purchases":{"entity":"purchases","quantity":"units","amount":"purchase_amount","category":"state"},
}},
{
"schema":{
"contractors":["contractor_id","contractor_name","division","hourly_pay","worker_age","career_experience"],
"catalog_items":["catalog_id","item_name","segment","sale_price","inventory_count","review_score"],
"shipments":["shipment_id","catalog_id","contractor_id","item_count","shipment_value","shipment_status"],
},
"profiles":{
"contractors":{"entity":"contractors","name":"contractor_name","category":"division","pay":"hourly_pay","age":"worker_age","experience":"career_experience"},
"catalog_items":{"entity":"catalog items","name":"item_name","category":"segment","price":"sale_price","stock":"inventory_count","rating":"review_score"},
"shipments":{"entity":"shipments","quantity":"item_count","amount":"shipment_value","category":"shipment_status"},
}},
{
"schema":{
"technicians":["tech_id","tech_name","unit","base_pay","current_age","field_experience"],
"merchandise":["sku","product_name","product_type","retail_price","stock_level","customer_rating"],
"sales":["sale_id","sku","tech_id","quantity_sold","sale_total","sale_status"],
},
"profiles":{
"technicians":{"entity":"technicians","name":"tech_name","category":"unit","pay":"base_pay","age":"current_age","experience":"field_experience"},
"merchandise":{"entity":"merchandise","name":"product_name","category":"product_type","price":"retail_price","stock":"stock_level","rating":"customer_rating"},
"sales":{"entity":"sales","quantity":"quantity_sold","amount":"sale_total","category":"sale_status"},
}},
{
"schema":{
"advisors":["advisor_id","advisor_name","practice","fee_rate","advisor_age","work_experience"],
"services":["service_id","service_name","service_group","service_price","capacity","service_rating"],
"requests":["request_id","service_id","advisor_id","requested_units","request_total","request_state"],
},
"profiles":{
"advisors":{"entity":"advisors","name":"advisor_name","category":"practice","pay":"fee_rate","age":"advisor_age","experience":"work_experience"},
"services":{"entity":"services","name":"service_name","category":"service_group","price":"service_price","stock":"capacity","rating":"service_rating"},
"requests":{"entity":"requests","quantity":"requested_units","amount":"request_total","category":"request_state"},
}},
{
"schema":{
"fleet_assets":["asset_id","asset_name","asset_class","asset_price","mileage","asset_rating"],
"operators":["operator_id","operator_name","region","operator_age","years_experience","daily_pay"],
"jobs":["job_id","asset_id","operator_id","job_units","job_amount","job_status"],
},
"profiles":{
"fleet_assets":{"entity":"fleet assets","name":"asset_name","category":"asset_class","price":"asset_price","mileage":"mileage","rating":"asset_rating"},
"operators":{"entity":"operators","name":"operator_name","category":"region","age":"operator_age","experience":"years_experience","pay":"daily_pay"},
"jobs":{"entity":"jobs","quantity":"job_units","amount":"job_amount","category":"job_status"},
}},
{
"schema":{
"agents":["agent_id","agent_name","branch","commission_rate","agent_age","industry_experience"],
"goods":["goods_id","goods_name","goods_category","goods_price","available_stock","goods_rating"],
"deliveries":["delivery_id","goods_id","agent_id","delivery_quantity","delivery_amount","delivery_status"],
},
"profiles":{
"agents":{"entity":"agents","name":"agent_name","category":"branch","pay":"commission_rate","age":"agent_age","experience":"industry_experience"},
"goods":{"entity":"goods","name":"goods_name","category":"goods_category","price":"goods_price","stock":"available_stock","rating":"goods_rating"},
"deliveries":{"entity":"deliveries","quantity":"delivery_quantity","amount":"delivery_amount","category":"delivery_status"},
}},
{
"schema":{
"analysts":["analyst_id","analyst_name","desk","compensation","person_age","domain_experience"],
"listings":["listing_id","listing_name","listing_type","asking_price","available_units","listing_rating"],
"invoices":["invoice_id","listing_id","analyst_id","invoice_quantity","invoice_amount","invoice_status"],
},
"profiles":{
"analysts":{"entity":"analysts","name":"analyst_name","category":"desk","pay":"compensation","age":"person_age","experience":"domain_experience"},
"listings":{"entity":"listings","name":"listing_name","category":"listing_type","price":"asking_price","stock":"available_units","rating":"listing_rating"},
"invoices":{"entity":"invoices","quantity":"invoice_quantity","amount":"invoice_amount","category":"invoice_status"},
}},
]

HELD_OUT_SCHEMAS={
"consultants":{
"consultants":["consultant_id","consultant_name","work_experience","hourly_rate"],
"departments":["department_id","department_name","budget"],
},
"vehicles":{
"vehicles":["id","brand","model","year","price","mileage","fuel_type","rating"],
"owners":["id","name","age","membership_level"],
},
}

ROLE_TEMPLATES={
"experience":[
("show records with experience more than 5 years",">","5"),
("people having over 5 years experience",">","5"),
("show records with work experience more than 10 years",">","10"),
("find entries with experience below 3 years","<","3"),
("records with exactly 7 years experience","=","7"),
],
"age":[
("show records older than 30",">","30"),
("show all records older than 15",">","15"),
("show people older than 15",">","15"),
("find entries younger than 20","<","20"),
("records aged exactly 30","=","30"),
],
"price":[
("show records priced below 20000","<","20000"),
("show records priced above 50",">","50"),
("find entries costing less than 100","<","100"),
("records with price exactly 75","=","75"),
],
"mileage":[
("show records with mileage greater than 30000",">","30000"),
("show records with mileage below 50000","<","50000"),
("records with mileage exactly 10000","=","10000"),
],
"rating":[
("show records with rating above 4.5",">","4.5"),
("find entries with rating below 3","<","3"),
("records rated exactly 4","=","4"),
],
"pay":[
("show records with pay greater than 60000",">","60000"),
("find entries earning less than 50000","<","50000"),
("records with pay exactly 45000","=","45000"),
],
"stock":[
("show records with stock greater than 10",">","10"),
("find entries with stock below 5","<","5"),
("records with stock exactly 20","=","20"),
],
"quantity":[
("show records with quantity greater than 2",">","2"),
("find entries containing more than 2 items",">","2"),
("records with quantity equal to 2","=","2"),
],
"amount":[
("show records worth more than 100",">","100"),
("find entries below amount 50","<","50"),
("records with amount exactly 75","=","75"),
],
}

LEGACY_ANCHORS=[
("show employees where salary is greater than 60000","SELECT * FROM employees WHERE salary > 60000;"),
("retrieve the full product list","SELECT * FROM products;"),
("which order statuses occur","SELECT DISTINCT status FROM orders;"),
("rank employees from lowest salary to highest","SELECT * FROM employees ORDER BY salary ASC;"),
("arrange orders from smallest quantity to largest","SELECT * FROM orders ORDER BY quantity ASC;"),
("arrange products from most expensive to cheapest","SELECT * FROM products ORDER BY price DESC;"),
("arrange orders from largest quantity to smallest","SELECT * FROM orders ORDER BY quantity DESC;"),
("employees aged exactly 30","SELECT * FROM employees WHERE age = 30;"),
("find employees under 30 years old","SELECT * FROM employees WHERE age < 30;"),
("orders containing more than 2 items","SELECT * FROM orders WHERE quantity > 2;"),
("return only 5 employees","SELECT * FROM employees LIMIT 5;"),
("retrieve 5 product records","SELECT * FROM products LIMIT 5;"),
("give me 5 order records","SELECT * FROM orders LIMIT 5;"),
("show employee names and salaries","SELECT name, salary FROM employees;"),
("show product names and prices","SELECT name, price FROM products;"),
("show order quantities and statuses","SELECT quantity, status FROM orders;"),
("show employee names and departments","SELECT name, department FROM employees;"),
("show employee names and salaries where age is greater than 30","SELECT name, salary FROM employees WHERE age > 30;"),
("show employee names and salaries where age equals 30","SELECT name, salary FROM employees WHERE age = 30;"),
("show product names and prices where price is less than 100","SELECT name, price FROM products WHERE price < 100;"),
("show order quantities and statuses where quantity is greater than 2","SELECT quantity, status FROM orders WHERE quantity > 2;"),
]

def _add(out,schema,q,sql,category):
    out.append({"schema":schema,"question":q,"sql":sql,"category":category})

def _structural(out,schema,table,p):
    entity=p["entity"]
    for q in (f"show all {entity}",f"list every {entity}",f"retrieve the full {entity} list"):
        _add(out,schema,q,f"SELECT * FROM {table};","select_all")
    for q in (f"how many {entity} are there",f"count the {entity}",f"give me the {entity} count"):
        _add(out,schema,q,f"SELECT COUNT(*) FROM {table};","count")
    for q in (f"return only 5 {entity}",f"show first 5 {entity}",f"retrieve 5 {entity} records"):
        _add(out,schema,q,f"SELECT * FROM {table} LIMIT 5;","limit")

    if "name" in p:
        for q in (f"show {entity} names",f"list names for {entity}"):
            _add(out,schema,q,f"SELECT {p['name']} FROM {table};","projection")

    if "category" in p:
        c=p["category"]
        for q in (f"show distinct {entity} categories",f"list unique {c.replace('_',' ')} values for {entity}",f"which {c.replace('_',' ')} values occur"):
            _add(out,schema,q,f"SELECT DISTINCT {c} FROM {table};","distinct")

    numeric=next((r for r in ("pay","price","amount","quantity","rating","mileage","stock","age") if r in p),None)
    if numeric:
        c=p[numeric]
        for q in (f"what is the average {numeric} for {entity}",f"find the mean {numeric} for {entity}"):
            _add(out,schema,q,f"SELECT AVG({c}) FROM {table};","avg")
        for q in (f"sort {entity} by {numeric} ascending",f"rank {entity} from lowest {numeric} to highest"):
            _add(out,schema,q,f"SELECT * FROM {table} ORDER BY {c} ASC;","order_asc")
        for q in (f"sort {entity} by {numeric} descending",f"rank {entity} from highest {numeric} to lowest"):
            _add(out,schema,q,f"SELECT * FROM {table} ORDER BY {c} DESC;","order_desc")
        if numeric=="price":
            _add(out,schema,f"arrange {entity} from most expensive to cheapest",f"SELECT * FROM {table} ORDER BY {c} DESC;","order_desc")
            _add(out,schema,f"arrange {entity} from cheapest to most expensive",f"SELECT * FROM {table} ORDER BY {c} ASC;","order_asc")

    if "name" in p:
        second=next((r for r in ("pay","price","age","rating","category") if r in p),None)
        if second:
            a,b=p["name"],p[second]
            for q in (f"show {entity} names and {second}",f"list {entity} names together with {second}"):
                _add(out,schema,q,f"SELECT {a}, {b} FROM {table};","multi_projection")

def _filters_and_compositions(out,schema,table,p):
    entity=p["entity"]
    for role,templates in ROLE_TEMPLATES.items():
        if role not in p:continue
        col=p[role]
        for q,op,value in templates:
            # Mix generic and entity-specific paraphrases.
            _add(out,schema,q,f"SELECT * FROM {table} WHERE {col} {op} {value};",f"where_{role}")
            _add(out,schema,q.replace("records",entity),f"SELECT * FROM {table} WHERE {col} {op} {value};",f"where_{role}")

    role=next((r for r in ("age","price","quantity") if r in p),None)
    if role:
        col=p[role]
        _add(out,schema,f"show {entity} where {role} is 25 or 30",f"SELECT * FROM {table} WHERE {col} IN (25, 30);",f"in_{role}")

    if "name" in p and "age" in p:
        _add(out,schema,f"show {entity} names where age is greater than 30",f"SELECT {p['name']} FROM {table} WHERE {p['age']} > 30;","projection_where")

    # Multi-column projection + WHERE anchors on generic schema roles.
    if "name" in p:
        second=next((r for r in ("pay","price","category","rating") if r in p),None)
        filter_role=next((r for r in ("age","price","quantity","amount","stock") if r in p),None)
        if second and filter_role:
            a,b=p["name"],p[second]
            fcol=p[filter_role]
            _add(out,schema,f"show {entity} names and {second} where {filter_role} is greater than 30",f"SELECT {a}, {b} FROM {table} WHERE {fcol} > 30;","multi_projection_where")

    # Compositions deliberately use different WHERE and ORDER columns when possible.
    roles=[r for r in ("age","experience","price","mileage","rating","pay","stock","quantity","amount") if r in p]
    if roles:
        w=roles[0]
        o=roles[1] if len(roles)>1 else roles[0]
        wc,oc=p[w],p[o]
        _add(out,schema,f"show {entity} where {w} is greater than 10 ordered by {o} descending",f"SELECT * FROM {table} WHERE {wc} > 10 ORDER BY {oc} DESC;","where_order")
        _add(out,schema,f"show {entity} where {w} is greater than 10 limited to 5",f"SELECT * FROM {table} WHERE {wc} > 10 LIMIT 5;","where_limit")
        _add(out,schema,f"show {entity} ordered by {o} descending limited to 5",f"SELECT * FROM {table} ORDER BY {oc} DESC LIMIT 5;","order_limit")
        _add(out,schema,f"show {entity} where {w} is greater than 10 ordered by {o} descending limited to 5",f"SELECT * FROM {table} WHERE {wc} > 10 ORDER BY {oc} DESC LIMIT 5;","where_order_limit")
        _add(out,schema,f"show {entity} where {w} is less than 20 ordered by {o} ascending limited to 5",f"SELECT * FROM {table} WHERE {wc} < 20 ORDER BY {oc} ASC LIMIT 5;","where_order_limit")

def _permute_schema(schema,rng):
    items=list(schema.items());rng.shuffle(items)
    result={}
    for table,columns in items:
        cols=list(columns);rng.shuffle(cols)
        result[table]=cols
    return result

def build_schema_conditioned_examples(seed=42,target_size=2942):
    base=[]
    for db in TRAIN_DATABASES:
        schema=db["schema"]
        for table,p in db["profiles"].items():
            _structural(base,schema,table,p)
            _filters_and_compositions(base,schema,table,p)

    # Preserve uploaded training examples under the complete legacy schema.
    try:
        from training.text_to_sql_dataset import TEXT_TO_SQL_EXAMPLES
        for example in TEXT_TO_SQL_EXAMPLES:
            _add(base,LEGACY_SCHEMA,example["question"],example["sql"],"legacy_source")
    except Exception:
        pass

    # Repeat verified legacy anchors as regression-preservation supervision.
    for _ in range(8):
        for q,sql in LEGACY_ANCHORS:
            _add(base,LEGACY_SCHEMA,q,sql,"legacy_anchor")

    rng=random.Random(seed)
    out=[]
    index=0
    while len(out)<target_size:
        example=base[index%len(base)]
        schema=_permute_schema(example["schema"],random.Random(seed+index*37))
        out.append({**example,"schema":schema})
        index+=1
    rng.shuffle(out)
    return out
