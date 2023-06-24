import psycopg2
import matplotlib.pyplot as plt

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_user",
    password="your_password"
)

# Create a cursor object
cursor = conn.cursor()

# Execute an SQL query to retrieve the data from the column
cursor.execute("SELECT column_name FROM your_table")

# Fetch all the data from the query result
data = cursor.fetchall()

# Close the cursor and the database connection
cursor.close()
conn.close()

# Extract the values from the data (assuming a single column result)
values = [row[0] for row in data]

# Create a histogram to visualize the distribution
plt.hist(values, bins=10)  # Adjust the number of bins as needed

# Customize the graph labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Column Data")

# Display the graph
plt.show()
