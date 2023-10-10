import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import community
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

output_folder = 'C:/Users/Asus/Desktop/Project/Community/Template'

def save_graph(graph, filename):
    # Create the full file path
    file_path = os.path.join(output_folder, filename)
    
    # Save the graph to the file
    graph.savefig(file_path)
    print(f"Saved graph to {filename}")

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('articles1.csv')
df1 = pd.read_csv('articles2.csv')

# Select the 3rd and 6th columns by column index (0-based index)
new_df = df.iloc[:, [2, 5]]
new_df1 = df1.iloc[:, [2, 5]]

new_dffinal = pd.concat([new_df, new_df1])

print("Number of rows in the dataset is :", new_dffinal.shape[0])
print("Number of columns in the dataset is :",new_dffinal.shape[1])
print(new_df)

new_dffinal['date'] = pd.to_datetime(new_dffinal['date'])


# Sort the DataFrame by 'date' in ascending order and limit the data to 600 rows
sorted_df = new_dffinal.sort_values(by='date')
# sorted_df = sorted_df.sample(frac=1, random_state=42)
sorted_df = new_dffinal.sort_values(by='date').head(120)

print(sorted_df)
print("Columns in this sorted dataset is : ")
print(sorted_df.columns)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.85,
    min_df=1,
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80 # Adjust this number as needed
)

# Drop NaN values in the 'title' column and create the TF-IDF matrix
sorted_df = sorted_df.dropna(subset=['title'])
tfidf_matrix = tfidf_vectorizer.fit_transform(sorted_df['title'])


print("The Tokenization scores is\n")
print(tfidf_matrix)
# Create a DataFrame from the non-zero elements
rows, cols = tfidf_matrix.nonzero()
values = [tfidf_matrix[row, col] for row, col in zip(rows, cols)]
data = {"Row": rows, "Column": cols, "Value": values}
df_token = pd.DataFrame(data)
df_token.set_index(['Row', 'Column'], inplace=True)
feature_names = tfidf_vectorizer.get_feature_names_out()
df_token['Feature'] = [feature_names[col] for col in df_token.index.get_level_values('Column')]
df_token = df_token.sort_values(by='Value', ascending=True)

# Group by feature and calculate mean TF-IDF values
df_token1 = df_token.groupby('Feature')['Value'].mean()

# Define a threshold for bursty terms
threshold = 0.7
bursty_terms = df_token1[df_token1 > threshold].index
less_bursty_terms = df_token1[df_token1 <= threshold].index

print(df_token)
print("The info of it is\n")
print(df_token.info())
df_token['Feature'].drop_duplicates(keep='last', inplace=True)
df_token =df_token[df_token['Feature'].apply(lambda x: not str(x).replace('.', '', 1).isdigit())]
print(df_token)

df_tokengraph = df_token
# df_token.plot(kind = 'scatter', x = 'Value', y = 'Feature')

# Create a graph
G = nx.Graph()

# Create a dictionary to store node colors based on 'Value'
node_colors = {}

# Assign colors based on 'Value' ranges
for index, row in df_token.iterrows():
    feature = row['Feature']
    value = row['Value']
    G.add_node(feature)
    
    if 0 <= value < 0.1:
        color = '#FF0000'  # Red
    elif 0.1 <= value < 0.2:
        color = '#FF3300'  # Darker Red
    elif 0.2 <= value < 0.3:
        color = '#FF6600'  # Even Darker Red
    elif 0.3 <= value < 0.4:
        color = '#FF9900'  # Even Darker Red
    elif 0.4 <= value < 0.5:
        color = '#FFCC00'  # Even Darker Red
    elif 0.5 <= value < 0.6:
        color = '#FFFF00'  # Yellow
    elif 0.6 <= value < 0.7:
        color = '#FFCC00'  # Even Darker Red
    elif 0.7 <= value < 0.8:
        color = '#FF9900'  # Even Darker Red
    elif 0.8 <= value < 0.9:
        color = '#FF6600'  # Even Darker Red
    else:
        color = '#FF3300'  # Darker Red
    
    node_colors[feature] = color

#Add edges based on node colors
for feature1 in G.nodes():
    for feature2 in G.nodes():
        if feature1 != feature2 and node_colors[feature1] == node_colors[feature2]:
            G.add_edge(feature1, feature2)

# Apply the Louvain algorithm for community detection
partition = community.best_partition(G)
# Create legend labels and handles based on partition
legend_labels = {}
for com in set(partition.values()):
    legend_labels[f'Community {com}'] = list(node_colors.values())[com]

legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
                  for label, color in legend_labels.items()]

# Add the legend to the graph with handles explicitly specified
plt.legend(handles=legend_handles, loc='upper right')

# Customize the graph layout and appearance
pos = nx.spring_layout(G, seed=42, k=0.7)

# Draw the graph with customizationsr
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=800, node_color=list(node_colors.values()),
        font_size=6, font_color='black',linewidths=1.5, edgecolors='black')


plt.title("Graph with Louvain Communities Colored by Value Intensity", fontsize=6)
plt.axis('off')
save_graph(plt, "Community_Detected.png")
plt.show()

threshold = 0.8  # Example threshold
df_token['Label'] = df_token.apply(lambda row: 1 if row['Value'] > threshold else 0, axis=1)
df_token.plot(kind = 'scatter', x = 'Feature', y = 'Label')
file_path = "C:/Users/Asus/Desktop/Project/Community/Template/scatter_plot.png"
plt.savefig(file_path)


G = nx.Graph()

# Create a dictionary to store node colors based on 'Value'
node_colors = {}

# Assign colors based on 'Value' ranges
for index, row in df_token.iterrows():
    feature = row['Feature']
    value = row['Label']
    G.add_node(feature)
    
    if value == 0:
        color = 'yellow'  # Red
    else:
        color = '#FF3300'  # Pink
    
    node_colors[feature] = color

#Add edges based on node colors
for feature1 in G.nodes():
    for feature2 in G.nodes():
        if feature1 != feature2 and node_colors[feature1] == node_colors[feature2]:
            G.add_edge(feature1, feature2)

# Apply the Louvain algorithm for community detection
partition = community.best_partition(G)

# Customize the graph layout and appearance
pos = nx.spring_layout(G, seed=42, k=0.7)

# Draw the graph with customizationsr
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=1000, node_color=list(node_colors.values()),
        font_size=6, font_color='black',linewidths=1.5, edgecolors='black')


plt.title("Graph with Bursty Terms Colored by Value Intensity", fontsize=6)
plt.axis('off')
save_graph(plt, "Event_Detected.png")
plt.show()

label_encoder = LabelEncoder()

# Assuming 'your_column' contains string data
df_token['Feature'] = label_encoder.fit_transform(df_token['Feature'])

scaler = MinMaxScaler()

# Assuming 'your_column' contains the data you want to normalize
data1 = df_token['Feature'].values.reshape(-1, 1)  # Reshape to a 2D array

# Fit and transform the data
normalized_data = scaler.fit_transform(data1)

# Update the column in your DataFrame with the normalized data
df_token['Feature'] = normalized_data

# Features and labels
X =df_token[['Value','Feature']]
y = df_token['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
save_graph(plt, "Confusion_Matrix.png")
plt.show()




   