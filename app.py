import streamlit as st

from text_to_sql_runtime import TextToSQLRuntime


st.set_page_config(page_title="Text-to-SQL GPT", page_icon="🗄️", layout="wide")


@st.cache_resource
def load_runtime():
    return TextToSQLRuntime()


runtime = load_runtime()


with st.sidebar:
    st.header("Database Schema")

    if st.button("Refresh Schema"):
        refreshed_schema = runtime.refresh_schema()
        st.success(
            f"Schema refreshed. {len(refreshed_schema)} table(s) found."
        )

    for table_name, columns in runtime.schema.items():
        with st.expander(table_name):
            for column in columns:
                st.write(column)


st.title("Text-to-SQL GPT")
st.caption("Natural language → SQL → database results")

question = st.text_input("Ask a question",placeholder="Example: show all employees")

if st.button("Generate SQL", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            sql = runtime.generate(question)
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
            rows = runtime.execute(sql)
            st.subheader("Results")
            if rows:
                try:
                    st.dataframe(rows, use_container_width=True)
                except (ImportError, ModuleNotFoundError):
                    st.json(rows)
            else:
                st.info("Query executed successfully but returned no rows.")

        except Exception as error:
            st.error(f"Unable to process the question: {error}")